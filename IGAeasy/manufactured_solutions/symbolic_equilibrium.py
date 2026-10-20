"""Differentiate continuum fluxes without expanding repeated geometry expressions."""
import sympy as sp


def _expression_jets(expressions, coordinates, order):
    replacements, outputs = sp.cse(expressions, symbols=sp.numbered_symbols('_g'))
    output_nodes = [sp.Symbol(f'_f{i}', real=True) for i in range(len(outputs))]
    definitions, derivatives, jets = [], {}, {}

    def derivative(expression, axis):
        value = sp.diff(expression, coordinates[axis])
        for symbol in sorted(expression.free_symbols, key=str):
            if (symbol, axis) in derivatives:
                value += sp.diff(expression, symbol)*derivatives[symbol, axis]
        return value

    # CSE nodes form a dependency graph. Apply the chain rule to each node once.
    for node, expression in [*replacements, *zip(output_nodes, outputs)]:
        du, dv = derivative(expression, 0), derivative(expression, 1)
        values = {(0, 0): expression, (1, 0): du, (0, 1): dv}
        if order == 2:
            values.update({(2, 0): derivative(du, 0), (1, 1): derivative(du, 1),
                           (0, 2): derivative(dv, 1)})
        symbols = {key: (node if key == (0, 0) else
                    sp.Symbol(f'{node}_{key[0]}{key[1]}', real=True)) for key in values}
        definitions.extend((symbols[key], value) for key, value in values.items())
        derivatives[node, 0], derivatives[node, 1] = symbols[1, 0], symbols[0, 1]
        if order == 2:
            derivatives[symbols[1, 0], 0] = symbols[2, 0]
            derivatives[symbols[1, 0], 1] = symbols[1, 1]
            derivatives[symbols[0, 1], 0] = symbols[1, 1]
            derivatives[symbols[0, 1], 1] = symbols[0, 2]
        jets[node] = symbols
    return definitions, [jets[node] for node in output_nodes]


def boundary_force_function(coordinates, area, tangents, first_flux, second_flux=None):
    """Compile the uncondensed physical force P_i - div(J*H_ji)/J."""
    expressions = [area, *tangents[0], *tangents[1]]
    expressions.extend(area*first_flux[i][k] for i in range(2) for k in range(3))
    if second_flux is not None:
        expressions.extend(area*second_flux[j][i][k]
                           for j in range(2) for i in range(2) for k in range(3))
    definitions, jets = _expression_jets(expressions, coordinates, 1)
    force = [jets[7+3*i+k][0, 0] for i in range(2) for k in range(3)]
    if second_flux is not None:
        for i in range(2):
            for k in range(3):
                force[3*i+k] -= jets[13+3*i+k][1, 0]+jets[19+3*i+k][0, 1]
    output = [jets[i][0, 0] for i in range(1, 7)]+[value/jets[0][0, 0] for value in force]
    return sp.lambdify(coordinates, output, 'numpy', cse=lambda _: (definitions, output), docstring_limit=0)


def equilibrium_function(coordinates, area, first_flux, director_value, director_flux,
                         second_flux=None):
    """Compile f=(-div(J*P)+div(div(J*H)))/J and director equilibrium.

    Fluxes are physical continuum quantities, before multiplication by area J.
    first_flux has shape (2,3), second_flux (2,2,3), director_value (2,),
    and director_flux (2,2). No finite element data enters this calculation.
    """
    expressions = [area]
    expressions.extend(area*first_flux[i][k] for i in range(2) for k in range(3))
    expressions.extend(area*director_value[k] for k in range(2))
    expressions.extend(area*director_flux[i][k] for i in range(2) for k in range(2))
    if second_flux is not None:
        expressions.extend(area*second_flux[i][j][k]
                           for i in range(2) for j in range(2) for k in range(3))
    definitions, jets = _expression_jets(expressions, coordinates, 2 if second_flux is not None else 1)
    J = jets[0][0, 0]
    body = [-jets[1+k][1, 0]-jets[4+k][0, 1] for k in range(3)]
    if second_flux is not None:
        for k in range(3):
            body[k] += (jets[13+k][2, 0]+jets[16+k][1, 1]
                        +jets[19+k][1, 1]+jets[22+k][0, 2])
    body.extend(jets[7+k][0, 0]-jets[9+k][1, 0]-jets[11+k][0, 1] for k in range(2))
    body = [value/J for value in body]
    # Supply the existing graph directly; re-expanding it would undo the savings.
    return sp.lambdify(coordinates, body, 'numpy',
                       cse=lambda _: (definitions, body), docstring_limit=0)
