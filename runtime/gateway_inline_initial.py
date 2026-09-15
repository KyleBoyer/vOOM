"""Opt-in initial real-tool catalog; ranking is availability, never execution."""
from .toolcalls import pinned_tool_indices, rank_tool_indices

FLAG = 'VMODEL_FAST_TOOL_GATEWAY_INLINE_INITIAL'
POLICY = (
    'The supplied real functions are available now. When the user requests a '
    'lookup or action involving external state, obtain it by calling the appropriate '
    'real function immediately as your first output, with no surrounding prose. '
    'Use vmodel_search_tools instead only when none of the supplied real functions '
    'can perform the requested operation. Do not enable functions already supplied. '
    'Do not invent external results, supply example results as real data, or promise '
    'to call a function later. If no external information or action is needed, '
    'answer normally from the conversation and stable knowledge. Preserve the '
    'caller instructions and exact requested identifiers, arguments and output format.'
)


def candidates(value, *, messages, tools, raw_tools, client_choice, force_reason,
               structured_output, host_route, terminal_synthesis, buffered,
               activated_names):
    if value not in ('0', '1'):
        raise ValueError(FLAG + ' must be 0 or 1')
    if value == '0':
        return None
    if (client_choice != 'auto'
            or force_reason not in (None, 'external-action-imperative')
            or structured_output is not None or host_route or terminal_synthesis
            or buffered != '1' or activated_names or not messages
            or messages[-1].get('role') != 'user'
            or any(m.get('role') == 'tool' or m.get('tool_calls') for m in messages)
            or not tools or len(raw_tools) != len(tools)):
        return None
    query = [messages[-1]]
    pinned = pinned_tool_indices(tools, query)
    if len(pinned) > 4:
        return None
    ranking = rank_tool_indices(tools, query, use_embeddings=False)
    selected = list(dict.fromkeys([*pinned, *ranking]))[:4]
    chosen, raw = [tools[i] for i in selected], [raw_tools[i] for i in selected]
    names = []
    for wrapped, original in zip(chosen, raw):
        function = wrapped.get('function', wrapped)
        raw_function = original.get('function', {
            k: v for k, v in original.items() if k != 'type'})
        name = function.get('name')
        if (not isinstance(name, str) or not name or name.startswith('vmodel_')
                or function != raw_function or name in names):
            return None
        names.append(name)
    return chosen, raw, len(pinned)
