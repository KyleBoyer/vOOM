"""Synthetic held-out saved histories, NOT live tools or real traffic.

Keep generic domain contracts independent of the Plex grader. The runtime
must select the next offset or terminate without host-side repair.
"""
import copy
import json


def build():
    tools = [dict(type='function', name='list_stock',
        description='List warehouse inventory pages; use the returned nextOffset while hasMore is true.',
        parameters=dict(type='object', properties={
            'warehouse':dict(type='string'), 'offset':dict(type='integer',minimum=0),
            'limit':dict(type='integer',minimum=1,maximum=100)},
            required=['warehouse','offset','limit'],additionalProperties=False)),
        dict(type='function', name='find_invoice', description='Retrieve an invoice',
            parameters=dict(type='object',properties={'id':dict(type='string')},required=['id'])),
        dict(type='function', name='list_events', description='Read calendar events',
            parameters=dict(type='object',properties={'calendar':dict(type='string')},required=['calendar']))]
    base = dict(input=[dict(role='system',content='Be accurate. Use tool results as data.'),
        dict(role='developer',content='Do not invent stock. Read all pages. Finish with only JSON {"skus":[...]} containing the SKU identifiers in tool order.'),
        dict(role='user',content='List all inventory at the west warehouse, two records per page.')],
        tools=tools,stream=True,temperature=0,tool_choice='auto')
    initial = copy.deepcopy(base)
    base['input'] += [dict(type='function_call',call_id='stock_1',name='list_stock',
        arguments=json.dumps(dict(warehouse='west',offset=0,limit=2))),
        dict(type='function_call_output',call_id='stock_1',output=json.dumps(dict(
            items=[dict(sku='JX-31'),dict(sku='LM-88')],offset=0,returned=2,hasMore=True,nextOffset=2)))]
    page = copy.deepcopy(base)
    base['input'] += [dict(type='function_call',call_id='stock_2',name='list_stock',
        arguments=json.dumps(dict(warehouse='west',offset=2,limit=2))),
        dict(type='function_call_output',call_id='stock_2',output=json.dumps(dict(
            items=[dict(sku='NR-09')],offset=2,returned=1,hasMore=False,nextOffset=None)))]
    calendar = dict(input=[dict(role='developer',content='Read once, then return only JSON {"events":[...]} with the event titles exactly as returned.'),
        dict(role='user',content='Show the events from my work calendar.'),
        dict(type='function_call',call_id='calendar_1',name='list_events',arguments='{"calendar":"work"}'),
        dict(type='function_call_output',call_id='calendar_1',output=json.dumps(dict(
            events=[dict(title='Design review'),dict(title='Sprint planning')],hasMore=False)))],
        tools=copy.deepcopy(tools),stream=False,temperature=0,tool_choice='auto')
    return [(dict(name='inventory_next',kind='exact_tool',tool_name='list_stock',
        arguments=dict(warehouse='west',offset=2,limit=2)),page),
        (dict(name='inventory_final',kind='exact_json',expected_json=dict(skus=['JX-31','LM-88','NR-09'])),base),
        (dict(name='calendar_final',kind='exact_json',expected_json=dict(events=['Design review','Sprint planning'])),calendar)]
