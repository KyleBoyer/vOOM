"""Focused synthetic final-answer diagnostic, NOT an original harness replay.

Uses the original captured user text and one actual media tool schema, but a
short synthetic system/developer prompt, a JSON title-list format requirement,
two pre-supplied synthetic pages and greedy sampling. No live tool execution.
This does not alter the full workflow benchmark or its unchanged legacy rubric.
"""
import copy
import json

from tests.fixtures.plex_agent_profile import PLEX_MEDIA_TOOL, SYNTHETIC_PAGES
from tests.fixtures.plex_finite_media import respond, PAGE5_PROFILE


def build(capture):
    tools=[copy.deepcopy(t) for t in capture['tools']
           if t.get('name',t.get('function',{}).get('name'))==PLEX_MEDIA_TOOL]
    if len(tools)!=1:
        raise ValueError('requires exact captured media schema')
    users=[copy.deepcopy(m) for m in capture['input'] if m.get('role')=='user']
    if len(users)!=1:
        raise ValueError('requires one original user turn')
    messages=[dict(role='system',content='Answer accurately using the supplied tool results.'),
        dict(role='developer',content='Return only JSON with key titles: an alphabetically sorted array of matching titles. Preserve exact titles. Do not include excluded titles or explanations.'),
        *users]
    for offset in (0,5):
        arguments=dict(mediaType='all',limit=500,offset=offset)
        call_id='call_focused_page_'+str(offset)
        result=respond(dict(name=PLEX_MEDIA_TOOL,arguments=arguments),
                       SYNTHETIC_PAGES,profile=PAGE5_PROFILE)
        messages += [dict(type='function_call',call_id=call_id,name=PLEX_MEDIA_TOOL,
                          arguments=json.dumps(arguments)),
            dict(type='function_call_output',call_id=call_id,output=json.dumps(result))]
    request=dict(model='placeholder',input=messages,tools=tools,
                 stream=True,temperature=0.0,tool_choice='auto')
    case=dict(name='focused_plex_synthesis',kind='exact_json',stream=True,
              expected_json={'titles':['ALPHA_G','BRAVO_PG13','CHARLIE_TVY','DELTA_TVY7']})
    return case,request
