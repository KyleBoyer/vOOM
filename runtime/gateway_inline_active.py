"""Opt-in one-pass continuation catalog; no host action or answer selection."""

FLAG = 'VMODEL_FAST_TOOL_GATEWAY_INLINE_ACTIVE'
POLICY = (
    'The supplied real tools are already available alongside private catalog search. '
    'Answer the user directly and completely when the conversation and tool results '
    'suffice. Apply the user criteria to the returned data. If another external '
    'operation is necessary, call an available real tool directly. Use '
    'vmodel_search_tools only if a different capability is needed. Do not enable '
    'tools that are already available, repeat completed operations, invent external '
    'facts, or promise to perform an action later.'
)


def enabled(value, *, messages, tools, raw_tools, client_choice, force_reason,
            structured_output, host_route, terminal_synthesis, buffered):
    if value not in ('0', '1'):
        raise ValueError(FLAG + ' must be 0 or 1')
    if value == '0':
        return False
    if (client_choice != 'auto' or force_reason is not None
            or structured_output is not None or host_route or terminal_synthesis
            or buffered != '1' or not messages
            or messages[-1].get('role') != 'tool'
            or not 1 <= len(tools) <= 4 or len(raw_tools) != len(tools)):
        return False
    names = []
    for wrapped, raw in zip(tools, raw_tools):
        function = wrapped.get('function', wrapped)
        raw_function = raw.get('function', {
            key: value for key, value in raw.items() if key != 'type'})
        name = function.get('name')
        if (not isinstance(name, str) or not name or name.startswith('vmodel_')
                or raw_function != function):
            return False
        names.append(name)
    return len(set(names)) == len(names)
