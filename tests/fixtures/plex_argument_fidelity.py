"""Independent stricter capture predicate gate, never argument or rubric repair.

This oracle is specific to the pinned request and must not enter production.
The legacy score accepts a section-name substitute for a root-path predicate.
Require the exact path filter (or unfiltered retrieval), and forbid unrelated
restrictive filters that can silently lose qualifying rows on a real server.
"""

def check(calls):
    failures=[]
    for i,call in enumerate(calls):
        a=call.get('arguments')
        if call.get('name') not in ('plugin__plex__plex_list_library',
                                   'plugin__plex__plex_list_library_media') or not isinstance(a,dict):
            failures.append(dict(call=i,reason='unsupported-or-malformed-call'))
            continue
        for key in ('excludePlexLibrarySectionName','plexLibrarySectionName',
                    'plexLibrarySectionId','sectionName','sectionId','rootFolderPath',
                    'query','qualityProfileId','qualityProfileName','monitored','hasFile',
                    'status','year','minYear','maxYear','contentRating'):
            if a.get(key) is not None:
                failures.append(dict(call=i,reason='unrequested-restrictive-filter',field=key))
        if a.get('excludeRootFolderPath') not in (None,'/Kids/'):
            failures.append(dict(call=i,reason='wrong-root-literal'))
        if a.get('ratingOperator') is not None:
            kind=a.get('mediaType') or 'all'
            if a['ratingOperator']!='lte' or kind not in ('all','movie','show'):
                failures.append(dict(call=i,reason='wrong-rating-operation'))
            for media,field,expected in [('movie','movieRatingValue','PG-13'),
                                          ('show','showRatingValue','TV-Y7')]:
                if kind in ('all',media) and (a.get(field) or a.get('ratingValue'))!=expected:
                    failures.append(dict(call=i,reason='wrong-rating-threshold',field=field))
    return dict(passed=not failures,failures=failures,call_count=len(calls),
        scope='captured-request argument fidelity; unfiltered retrieval needs independent final-answer and coverage gates')
