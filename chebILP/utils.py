

def split_prolog_literals(body):
    """Split a Prolog rule body into literals, respecting parenthesis depth."""
    literals, current, depth = [], [], 0
    for char in body:
        if char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
        elif char == ',' and depth == 0:
            literals.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        literals.append(''.join(current).strip())
    return literals