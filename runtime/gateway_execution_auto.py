"""Opt-in model-authored answers after voluntary catalog activation.

No request content, tool names, rows, or domain-specific terminal predicates.
Required/specific client choices and forced external-action phases stay required.
"""

FLAG = 'VMODEL_FAST_TOOL_GATEWAY_EXECUTION_AUTO'
POLICY = (
    'Private execution phase: the selected real tools are available, not mandatory. '
    'If the conversation and tool results already answer the user, answer directly '
    'and completely from that evidence. Apply the user criteria to raw results. '
    'Call a provided tool only when more external information or action is needed. '
    'Do not invent external facts, promise a future action, or repeat completed calls.'
)


def enabled(value, *, client_choice, force_reason, structured_output, abstention_available):
    if value not in ('0', '1'):
        raise ValueError(FLAG + ' must be 0 or 1')
    return (value == '1' and client_choice == 'auto' and force_reason is None
        and not structured_output and abstention_available is True)


def plain_answer(content, calls):
    # An incomplete private/real call must not become ordinary visible prose.
    markers = ('<tool_call', '</tool_call', '<|tool_call', '[TOOL_CALLS]',
               '<function=', '<|python_tag|>', 'vmodel_no_suitable_tool')
    return (isinstance(content, str) and bool(content.strip()) and not calls
        and not any(marker in content for marker in markers))
