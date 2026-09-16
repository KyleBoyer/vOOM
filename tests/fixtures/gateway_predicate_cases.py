"""Synthetic counterbalanced path/label tasks; not real traffic or Plex data."""
import copy
import json


def build():
    tools=[dict(type='function',name='list_assets',description='Read file assets. Folder paths and collection labels are independent attributes.',
        parameters=dict(type='object',properties={
            'excludePath':dict(type='string',description='Exclude records whose folder path contains this exact substring.'),
            'excludeLabel':dict(type='string',description='Exclude records whose collection label equals this string.'),
            'limit':dict(type='integer'), 'offset':dict(type='integer')},additionalProperties=False)),
        dict(type='function',name='get_invoice',description='Read an invoice',parameters=dict(type='object',properties={'id':dict(type='string')}))]
    def request(user):
        return dict(input=[dict(role='system',content='Read-only tool use. Be accurate and preserve literal arguments.'),
            dict(role='user',content=user)],tools=copy.deepcopy(tools),temperature=1,stream=True,tool_choice='auto')
    path=request('List assets whose folder path does not contain "/Archive/". Fetch the first 20 records at offset 0. Do not filter collection labels.')
    label=request('List assets whose collection label is not "Archive". Fetch the first 20 records at offset 0. Do not filter folder paths.')
    final=request('List assets whose folder path does not contain "/Archive/". Only output JSON {"ids":[...]} in returned order. Collection labels do not matter.')
    final['stream']=False
    final['input'] += [dict(type='function_call',call_id='assets1',name='list_assets',arguments='{"limit":20,"offset":0}'),
        dict(type='function_call_output',call_id='assets1',output=json.dumps(dict(hasMore=False,filtersApplied=False,assets=[
            dict(id='FILE-7',folderPath='/Current/Design/',collectionLabel='Archive'),
            dict(id='FILE-9',folderPath='/Archive/Design/',collectionLabel='Current'),
            dict(id='FILE-2',folderPath='/Current/Photos/',collectionLabel='Current'),
            dict(id='FILE-4',folderPath='/Archive/Photos/',collectionLabel='Archive')])))]
    return [(dict(name='path_argument',kind='exact_tool',tool_name='list_assets',arguments=dict(excludePath='/Archive/',limit=20,offset=0)),path),
        (dict(name='label_argument',kind='exact_tool',tool_name='list_assets',arguments=dict(excludeLabel='Archive',limit=20,offset=0)),label),
        (dict(name='path_filter_final',kind='exact_json',expected_json=dict(ids=['FILE-7','FILE-2'])),final)]
