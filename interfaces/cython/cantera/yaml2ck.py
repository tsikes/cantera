# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

''' 
Adapted from Kyle Niemeyer's pyMARS Jul 24, 2019

Writes a solution object to a chemkin inp file
currently only works for Elementary, Falloff and ThreeBody Reactions
Cantera development version 2.3.0a2 required

KE Niemeyer, CJ Sung, and MP Raju. Skeletal mechanism generation for surrogate fuels using directed relation graph with error propagation and sensitivity analysis. Combust. Flame, 157(9):1760--1770, 2010. doi:10.1016/j.combustflflame.2009.12.022
KE Niemeyer and CJ Sung. On the importance of graph search algorithms for DRGEP-based mechanism reduction methods. Combust. Flame, 158(8):1439--1443, 2011. doi:10.1016/j.combustflflame.2010.12.010.
KE Niemeyer and CJ Sung. Mechanism reduction for multicomponent surrogates: A case study using toluene reference fuels. Combust. Flame, in press, 2014. doi:10.1016/j.combustflame.2014.05.001
TF Lu and CK Law. Combustion and Flame, 154:153--163, 2008. doi:10.1016/j.combustflame.2007.11.013

'''

import os, pathlib, sys
from textwrap import fill
from collections import Counter
import getopt

import cantera as ct

try:
    import ruamel_yaml as yaml
except ImportError:
    from ruamel import yaml

# number of calories in 1000 Joules
CALORIES_CONSTANT = 4184.0

# Conversion from 1 debye to coulomb-meters
DEBEYE_CONVERSION = 3.33564e-30

def reorder_reaction_equation(solution, reaction):
    # Split Reaction Equation
    rxn_eqn = reaction.equation
    for reaction_direction in [' <=> ', ' <= ', ' => ']:
        if reaction_direction in rxn_eqn:
            break
    for third_body in [' (+M)', ' + M', '']: # search eqn for third body
        if third_body in rxn_eqn:            # if reaches '', doesn't exist
            break

    # Sort and apply to reaction equation
    reaction_txt = []
    reaction_split = {'reactants': reaction.reactants, 
                      'products': reaction.products}
    for n, (reaction_side, species) in enumerate(reaction_split.items()):
        species_weights = []
        for key in species.keys():
            index = solution.species_index(key)
            species_weights.append(solution.molecular_weights[index])
        
        # Append coefficient to species
        species_list = []
        for species_text, coef in species.items():
            if coef == 1.0:
                species_list.append(species_text)
            else:
                species_list.append(f'{coef:.0f} {species_text}')
                
        species = species_list
        
        # Reorder species based on molecular weights
        species = [x for y, x in sorted(zip(species_weights, species))][::-1]
        reaction_txt.append(' + '.join(species) + third_body)
    
    reaction_txt = reaction_direction.join(reaction_txt)
    
    return reaction_txt
   

def eformat(f, precision=7, exp_digits=3):
    s = f"{f: .{precision}e}"
    if s == ' inf' or s == '-inf':
        return s
    else:
        mantissa, exp = s.split('e')
        exp_digits += 1 # +1 due to sign
        return f"{mantissa}E{int(exp):+0{exp_digits}}" 
  
 
def build_arrhenius(rate, reaction_order, reaction_type):
    """Builds Arrhenius coefficient string based on reaction type.
    Parameters
    ----------
    rate : cantera.Arrhenius
        Arrhenius-form reaction rate coefficient
    reaction_order : int or float
        Order of reaction (sum of reactant stoichiometric coefficients)
    reaction_type : {cantera.ElementaryReaction, cantera.ThreeBodyReaction, cantera.PlogReaction}
        Type of reaction
    Returns
    -------
    str
        String with Arrhenius coefficients
    """
    if reaction_type in [ct.ElementaryReaction, ct.PlogReaction]:
        pre_exponential_factor = rate.pre_exponential_factor * 1e3**(reaction_order - 1)

    elif reaction_type == ct.ThreeBodyReaction:
        pre_exponential_factor = rate.pre_exponential_factor * 1e3**reaction_order

    elif reaction_type in [ct.FalloffReaction, ct.ChemicallyActivatedReaction]:
        raise ValueError('Function does not support falloff or chemically activated reactions')
    else:
        raise NotImplementedError('Reaction type not supported: ', reaction_type)
    
    activation_energy = rate.activation_energy / CALORIES_CONSTANT
    arrhenius = [f'{eformat(pre_exponential_factor)}',
                 f'{eformat(rate.temperature_exponent)}', 
                 f'{eformat(activation_energy)}']
    return '   '.join(arrhenius)


def build_falloff_arrhenius(rate, reaction_order, reaction_type, pressure_limit):
    """Builds Arrhenius coefficient strings for falloff and chemically-activated reactions.
    Parameters
    ----------
    rate : cantera.Arrhenius
        Arrhenius-form reaction rate coefficient
    reaction_order : int or float
        Order of reaction (sum of reactant stoichiometric coefficients)
    reaction_type : {ct.FalloffReaction, ct.ChemicallyActivatedReaction}
        Type of reaction
    pressure_limit : {'high', 'low'}
        string designating pressure limit
    
    Returns
    -------
    str
        Arrhenius coefficient string
    """
    assert pressure_limit in ['low', 'high'], 'Pressure range needs to be high or low'

    # Each needs more complicated handling due if high- or low-pressure limit
    if reaction_type == ct.FalloffReaction:
        if pressure_limit == 'low':
            pre_exponential_factor = rate.pre_exponential_factor * 1e3**(reaction_order)
        elif pressure_limit == 'high':
            pre_exponential_factor = rate.pre_exponential_factor * 1e3**(reaction_order - 1)

    elif reaction_type == ct.ChemicallyActivatedReaction:
        if pressure_limit == 'low':
            pre_exponential_factor = rate.pre_exponential_factor * 1e3**(reaction_order - 1)
        elif pressure_limit == 'high':
            pre_exponential_factor = rate.pre_exponential_factor * 1e3**(reaction_order - 2)
    else:
        raise ValueError('Reaction type not supported: ', reaction_type)

    activation_energy = rate.activation_energy / CALORIES_CONSTANT
    arrhenius = [f'{eformat(pre_exponential_factor)}', 
                 f'{eformat(rate.temperature_exponent)}', 
                 f'{eformat(activation_energy)}'
                 ]
    return '   '.join(arrhenius)


def build_falloff(parameters, falloff_function):
    """Creates falloff reaction Troe parameter string
    Parameters
    ----------
    parameters : numpy.ndarray
        Array of falloff parameters; length varies based on ``falloff_function``
    falloff_function : {'Troe', 'SRI'}
        Type of falloff function
    Returns
    -------
    falloff_string : str
        String of falloff parameters
    """
    if falloff_function == 'Troe':
        falloff = [f'{eformat(f)}'for f in parameters]
        falloff_string = f"TROE / {'   '.join(falloff)} /\n"
    elif falloff_function == 'SRI':
        falloff = [f'{eformat(f)}'for f in parameters]
        falloff_string = f"SRI / {'   '.join(falloff)} /\n"
    else:
        raise NotImplementedError(f'Falloff function not supported: {falloff_function}')

    return falloff_string


def build_nasa7():
    # first line has species name, space for notes/date, elemental composition,
    # phase, thermodynamic range temperatures (low, high, middle), and a "1"
    # total length should be 80
    #
    # Ex: 
    # C6 linear biradi  T04/09C  6.   0.   0.   0.G   200.000  6000.000 1000.        1
    #  1.06841281E+01 5.62944075E-03-2.13152905E-06 3.56133777E-10-2.18273469E-14    2
    #  1.43741693E+05-2.87959136E+01 3.06949687E+00 3.71386246E-02-5.95698852E-05    3
    #  5.15924485E-08-1.77143386E-11 1.45477274E+05 8.35844575E+00 1.47610437E+05    4
    pass

def build_nasa9():
    # Ex: 
    #   therm NASA9
    #     200.00  1000.00  6000.00  20000.   3/19/02
    # e-                Ref-Species. Chase, 1998 3/82.                                
    #  3 912/98 E   1.00    0.00    0.00    0.00    0.00 0.000548579903          0.000
    #     298.150   1000.0007 -2.0 -1.0  0.0  1.0  2.0  3.0  4.0  0.0         6197.428
    #  0.000000000D+00 0.000000000D+00 2.500000000D+00 0.000000000D+00 0.000000000D+00
    #  0.000000000D+00 0.000000000D+00                -7.453750000D+02-1.172081224D+01
    #    1000.000   6000.0007 -2.0 -1.0  0.0  1.0  2.0  3.0  4.0  0.0         6197.428
    #  0.000000000D+00 0.000000000D+00 2.500000000D+00 0.000000000D+00 0.000000000D+00
    #  0.000000000D+00 0.000000000D+00                -7.453750000D+02-1.172081224D+01
    #    6000.000  20000.0007 -2.0 -1.0  0.0  1.0  2.0  3.0  4.0  0.0         6197.428
    #  0.000000000D+00 0.000000000D+00 2.500000000D+00 0.000000000D+00 0.000000000D+00
    #  0.000000000D+00 0.000000000D+00                -7.453750000D+02-1.172081224D+01
    #
    # Could build NASA9 from NASA7 by setting first 2 coefficients to 0
    pass


def species_data_text(solution_species):
    """Returns species declarations in Chemkin-format file.
    Parameters
    ----------
    solution_species : list of cantera.Species
        List of species objects
    """

    max_species_len = max([len(s.name) for s in solution_species])
    if any([s.note for s in solution_species]): # check that any notes exist
        max_species_len = max([16, max_species_len])
        species_txt = []
        for species in solution_species:
            text = f'{species.name:<{max_species_len}}   ! {species.note}\n'
            species_txt.append(text)
        
        species_txt = ''.join(species_txt)
        
    else:
        species_names = [f"{s.name:<{max_species_len}}" for s in solution_species]
        species_names = fill(
            '  '.join(species_names), 
            width=72,   # max length is 16, this gives 4 species per line
            break_long_words=False,
            break_on_hyphens=False
            )
        
        species_txt = f'{species_names}\n'
        
    text = ('SPECIES\n' + 
            species_txt + 
            'END\n\n\n')
    
    return text


def thermo_data_text(solution_species, input_type='included'): # Currently only NASA7, need to implement NASA9
    """Returns thermodynamic data in Chemkin-format file.
    Parameters
    ----------
    solution_species : list of cantera.Species
        List of species objects
    input_type : str, optional
        'included' if thermo will be printed in mech file, 'file' otherwise
    """
    
    if input_type == 'included':
        thermo_text = ['THERMO ALL\n' +  
                       '   300.000  1000.000  6000.000\n']
    else:
        thermo_text = ['THERMO\n' +  
                       '   300.000  1000.000  6000.000\n']

    # write data for each species in the Solution object
    for species in solution_species:
        composition_string = ''.join([f'{s:2}{int(v):>3}' 
                                      for s, v in species.composition.items()
                                      ])
        # attempt to split note and comment
        if len(species.note.split('\n', 1)) == 1:
            comment = ''
            comment_str = ''
            note_str = species.note
        else:
            comment = '!\n'
            note_str, comment_str = species.note.split('\n', 1)
        
        if len(f'{species.name} {note_str}') > 24:
            comment_str += '\n' + note_str
            note_str = ''
            
        comment_str = comment_str.replace('\n', '\n! ')
        comment = f'{comment}! {comment_str}'
        
        name_and_note = f'{species.name} {note_str}'
        species_string = (comment + '\n' +
            f'{name_and_note:<24}' + # name and date/note field
            f'{composition_string:<20}' +
            'G' + # only supports gas phase
            f'{species.thermo.min_temp:10.3f}' +
            f'{species.thermo.max_temp:10.3f}' +
            f'{species.thermo.coeffs[0]:10.3f}' +
            f'{1:>5}\n') # unused atomic symbols/formula, and blank space
        
        # second line has first five coefficients of high-temperature range,
        # ending with a "2" in column 79
        species_string += (
            ''.join([f'{c:15.8e}' for c in species.thermo.coeffs[1:6]]) +
            f'{2:>5}\n')
        
        # third line has the last two coefficients of the high-temperature range,
        # first three coefficients of low-temperature range, and "3"
        species_string += (
            ''.join([f'{c:15.8e}' for c in species.thermo.coeffs[6:8]]) +
            ''.join([f'{c:15.8e}' for c in species.thermo.coeffs[8:11]]) +
            f'{3:>5}\n')

        # fourth and last line has the last four coefficients of the
        # low-temperature range, and "4"
        species_string += (
            ''.join([f'{c:15.8e}' for c in species.thermo.coeffs[11:15]]) +
            f'{4:>20}\n')

        thermo_text.append(species_string)
    
    if input_type == 'included':
        thermo_text.append('END\n\n\n')
    else:
        thermo_text.append('END\n')
    
    return ''.join(thermo_text) 
    

def write_transport_data(species_list, filename='generated_transport.dat'): # need to add notes
    """Writes transport data to Chemkin-format file.
    Parameters
    ----------
    species_list : list of cantera.Species
        List of species objects
    filename : path or str, optional
        Filename for new Chemkin transport database file
    """
    geometry = {'atom': '0', 'linear': '1', 'nonlinear': '2'}
    with open(filename, 'w') as trans_file:

        # write data for each species in the Solution object
        for species in species_list:
            
            # each line contains the species name, integer representing
            # geometry, Lennard-Jones potential well depth in K,
            # Lennard-Jones collision diameter in angstroms,
            # dipole moment in Debye,
            # polarizability in cubic angstroms, and
            # rotational relaxation collision number at 298 K.
            species_string = (
                f'{species.name:<16}' +
                f'{geometry[species.transport.geometry]:>4}' +
                f'{(species.transport.well_depth / ct.boltzmann):>10.3f}' + 
                f'{(species.transport.diameter * 1e10):>10.3f}' + 
                f'{(species.transport.dipole / DEBEYE_CONVERSION):>10.3f}' + 
                f'{(species.transport.polarizability * 1e30):>10.3f}' + 
                f'{species.transport.rotational_relaxation:>10.3f}' + 
                '\n'
            )
            
            trans_file.write(species_string)


def write(solution, output_path='', reorder_reactions=True,
          skip_thermo=False, seperate_thermo_file=False, 
          skip_transport=False):
    """Writes Cantera solution object to Chemkin-format file.
    Parameters
    ----------
    solution : cantera.Solution
        Model to be written
    output_path : path or str, optional
        Path of file to be written; if not provided, use cd / 'solution.name'
    reorder_reactions : bool, optional
        Flag to reorder reactions based upon molecular weight
    skip_thermo : bool, optional
        Flag to skip writing thermo data
    seperate_thermo_file : bool, optional
        Flag to write thermo data in a seperate file from mechanism
    skip_transport : bool, optional
        Flag to skip writing transport data in separate file
    Returns
    -------
    output_file_name : list
        List of paths to output mechanism files (.ck, .therm, .tran)
    Examples
    --------
    >>> gas = cantera.Solution('gri30.cti')
    >>> soln2ck.write(gas)
    [output_path / reduced_gri30.ck, output_path / reduced_gri30.therm, 
     output_path / reduced_gri30.tran]
    """
    if output_path:
        if not isinstance(output_path, pathlib.PurePath):
            try:    # try to turn full path into pathlib Path
                output_path = pathlib.Path(output_path)
            except:   # if previous fails, assume str is file name
                output_path = pathlib.Path.cwd() / output_path
    else:
        output_path = pathlib.Path.cwd() / f'{solution.name}.ck'

    if output_path.is_dir():
        output_path = output_path / f'{solution.name}.ck'

    if output_path.is_file():
        output_path.unlink()
       
    main_path = output_path.parents[0]
    basename = output_path.stem
    output_files = [output_path]
        
    with open(output_path, 'w') as mech_file:
        # Write title block to file
        if solution.description:
            note = solution.description.replace('\n', '\n! ')
            mech_file.write(f'! {note}\n!\n')
        if '! Chemkin file converted from Cantera solution object' not in solution.description:
            mech_file.write('! Chemkin file converted from Cantera solution object\n! \n\n')

        # write species and element lists to file
        element_names = '  '.join(solution.element_names)
        mech_file.write(
            'ELEMENTS\n' + 
            f'{element_names}\n' +
            'END\n\n\n'
            )
        
        mech_file.write(species_data_text(solution.species()))

        # Write thermo to file
        if not skip_thermo and not seperate_thermo_file:
            mech_file.write(thermo_data_text(solution.species(), input_type='included'))
            
        # Write reactions to file
        max_rxn_width = 3 + max([len(rxn.equation) for rxn in solution.reactions()] + [48])
        
        mech_file.write('REACTIONS  CAL/MOLE  MOLES\n')
        # Write data for each reaction in the Solution Object
        for n, reaction in enumerate(solution.reactions()):
            reaction_string = ''

            if reaction.note:
                rxn_note = [f'! {note.strip()}' for note in reaction.note.rsplit('\n')]
                if len(rxn_note) > 1:
                    reaction_string += '\n'.join(rxn_note[:-1]) + '\n'
            
            if reorder_reactions:
                reaction_equation = reorder_reaction_equation(solution, reaction)
            else:
                reaction_equation = reaction.equation
            reaction_string += f'{reaction_equation:<{max_rxn_width}}'

            # The Arrhenius parameters that follow the equation string on the main line 
            # depend on the type of reaction.
            if type(reaction) in [ct.ElementaryReaction, ct.ThreeBodyReaction]:
                arrhenius = build_arrhenius(
                    reaction.rate, 
                    sum(reaction.reactants.values()), 
                    type(reaction)
                    )

            elif type(reaction) == ct.FalloffReaction:
                # high-pressure limit is included on the main reaction line
                arrhenius = build_falloff_arrhenius(
                    reaction.high_rate, 
                    sum(reaction.reactants.values()), 
                    ct.FalloffReaction,
                    'high'
                    )

            elif type(reaction) == ct.ChemicallyActivatedReaction:
                # low-pressure limit is included on the main reaction line
                arrhenius = build_falloff_arrhenius(
                    reaction.low_rate, 
                    sum(reaction.reactants.values()), 
                    ct.ChemicallyActivatedReaction,
                    'low'
                    )

            elif type(reaction) == ct.ChebyshevReaction:
                arrhenius = '1.0e0  0.0  0.0'

            elif type(reaction) == ct.PlogReaction:
                arrhenius = build_arrhenius(
                    reaction.rates[0][1],
                    sum(reaction.reactants.values()), 
                    ct.PlogReaction
                    )
            else:
                raise NotImplementedError(f'Unsupported reaction type: {type(reaction)}')

            reaction_string += f'{arrhenius}    {rxn_note[-1]}\n'
            
            # now write any auxiliary information for the reaction
            if type(reaction) == ct.FalloffReaction:
                # for falloff reaction, need to write low-pressure limit Arrhenius expression
                arrhenius = build_falloff_arrhenius(
                    reaction.low_rate, 
                    sum(reaction.reactants.values()), 
                    ct.FalloffReaction,
                    'low'
                    )
                reaction_string += f'{"LOW /   ".rjust(max_rxn_width)}{arrhenius} /\n'

                # need to print additional falloff parameters if present
                if reaction.falloff.parameters.size > 0:
                    falloff_str = build_falloff(reaction.falloff.parameters, reaction.falloff.type)
                    width = max_rxn_width - 10 - 15*(reaction.falloff.parameters.size - 3)
                    reaction_string += f'{"".ljust(width)}{falloff_str}'

            elif type(reaction) == ct.ChemicallyActivatedReaction:
                # for chemically activated reaction, need to write high-pressure expression
                arrhenius = build_falloff_arrhenius(
                    reaction.low_rate, 
                    sum(reaction.reactants.values()), 
                    ct.ChemicallyActivatedReaction,
                    'high'
                    )
                reaction_string += f'HIGH'
                reaction_string += f'{"HIGH /   ".rjust(max_rxn_width)}{arrhenius} /\n'

                # need to print additional falloff parameters if present
                if reaction.falloff.parameters.size > 0:
                    falloff_str = build_falloff(reaction.falloff.parameters, reaction.falloff.type)
                    width = max_rxn_width - 10 - 15*(reaction.falloff.parameters.size - 3)
                    reaction_string += f'{"".ljust(width)}{falloff_str}'

            elif type(reaction) == ct.PlogReaction:
                # just need one rate per line
                for rate in reaction.rates:
                    pressure = f'{eformat(rate[0] / ct.one_atm)}'
                    arrhenius = build_arrhenius(rate[1], 
                                                sum(reaction.reactants.values()), 
                                                ct.PlogReaction
                                                )
                    reaction_string += (f'{"PLOG / ".rjust(max_rxn_width-18)}'
                                        f'{pressure}   {arrhenius} /\n')

            elif type(reaction) == ct.ChebyshevReaction:
                reaction_string += (
                    f'TCHEB / {reaction.Tmin}  {reaction.Tmax} /\n' +
                    f'PCHEB / {reaction.Pmin / ct.one_atm}  {reaction.Pmax / ct.one_atm} /\n' +
                    f'CHEB / {reaction.nTemperature}  {reaction.nPressure} /\n'
                    )
                for coeffs in reaction.coeffs:
                    coeffs_row = ' '.join([f'{c:.6e}' for c in coeffs])
                    reaction_string += f'CHEB / {coeffs_row} /\n'
            
            # need to trim and print third-body efficiencies, if present
            if type(reaction) in [ct.ThreeBodyReaction, ct.FalloffReaction, 
                                  ct.ChemicallyActivatedReaction
                                  ]:
                # trims efficiencies list
                reduced_efficiencies = {s:reaction.efficiencies[s] 
                                        for s in reaction.efficiencies
                                        if s in solution.species_names
                                        }
                efficiencies_str = '  '.join([f'{s}/ {v:.3f}/' for s, v in reduced_efficiencies.items()])
                if efficiencies_str:
                    reaction_string += '   ' + efficiencies_str + '\n'
            
            if reaction.duplicate:
                reaction_string += '   DUPLICATE\n'
                                
            mech_file.write(reaction_string)

        mech_file.write('END')

    # write thermo data
    if not skip_thermo and seperate_thermo_file:
        therm_path = main_path / f'{basename}.therm'
        with open(therm_path, 'w') as thermo_file:
            thermo_file.write(thermo_data_text(solution.species(), input_type='file'))
        output_files.append(therm_path)

    # TODO: more careful check for presence of transport data?
    if not skip_transport and all(sp.transport for sp in solution.species()):
        trans_path = main_path / f'{basename}.tran'
        write_transport_data(solution.species(), trans_path)
        output_files.append(trans_path)

    return output_files


def convert_mech(input, output_path='', reorder_reactions=True,
                 skip_thermo=False, seperate_thermo_file=False, 
                 skip_transport=False):

    solution = ct.Solution(str(input))
    return write(solution, output_path, reorder_reactions,
                 skip_thermo, seperate_thermo_file, skip_transport)


def main(argv):

    longOptions = ['input=', 'output_path=', 'reorder_reactions=', 'skip_thermo', 
                   'seperate_thermo_file', 'skip_transport', 'help', 'debug',
                   'no-validate']

    try:
        optlist, args = getopt.getopt(argv, 'dh', longOptions)
        options = dict()
        for o,a in optlist:
            options[o] = a

        if args:
            raise getopt.GetoptError('Unexpected command line option: ' +
                                     repr(' '.join(args)))

    except getopt.GetoptError as e:
        print('yaml2ck.py: Error parsing arguments:')
        print(e)
        print('Run "yaml2ck.py --help" to see usage help.')
        sys.exit(1)

    if not options or '-h' in options or '--help' in options:
        print(__doc__)
        sys.exit(0)

    input = options.get('--input')
    output_path = options.get('--output_path').strip('"')
    if '--reorder_reactions' not in options:
        reorder_reactions = True
    else:
        true_str = ['true', '1', 't', 'y', 'yes']
        reorder_reactions = options.get('--reorder_reactions').lower() in true_str
    skip_thermo = '--skip_thermo' in options
    seperate_thermo_file = '--seperate_thermo_file' in options
    skip_transport = '--skip_transport' in options

    if input:
        solution = ct.Solution(str(input))
    else:
        print('An input yaml must be provided as an argument to "--input="')
        sys.exit(1)

    output_paths = write(solution, output_path, reorder_reactions,
                         skip_thermo, seperate_thermo_file, skip_transport)

    if '--no-validate' in options:
        return

    try:
        from cantera import ck2yaml
        import tempfile

        print('Validating mechanism...', end='')
        ck = {'mech': str(output_paths[0]), 'therm': None, 'tran': None}
        
        for path in output_paths:
            for key in ck.keys():
                if key in path.suffix:
                    ck[key] = str(path)
        
        tf = tempfile.NamedTemporaryFile(suffix='.yaml', prefix='test_mech', delete=False)
        ck2yaml.convert_mech(ck['mech'], thermo_file=ck['therm'], transport_file=ck['tran'],
                         phase_name='gas', out_name=tf.name, quiet=True, permissive=True)
        gas = ct.Solution(tf.name)

        # for surf_name in surfaces:
        #     phase = ct.Interface(out_name, surf_name, [gas])
        tf.close()
        os.remove(tf.name)
        print('PASSED.')
    except RuntimeError as e:
        print('FAILED.')
        print(e)
        tf.close()
        os.remove(tf.name)
        sys.exit(1)

def script_entry_point():
    main(sys.argv[1:])

if __name__ == '__main__':
    main(sys.argv[1:])