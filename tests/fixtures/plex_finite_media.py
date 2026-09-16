"""Finite, query-isolated synthetic catalog in the observed media envelope.

Unlike the legacy mixed two-page queue, offset/limit index actual records and
exhausted queries return no rows. Synthetic rootFolderPath fields deliberately
exercise the user's root predicate; the real Kai export lacked root evidence.
No model response is rewritten and no rating/location filtering is performed
by this mock: filtersApplied=False makes that limitation explicit to the model.
"""
import copy

PROFILE = 'synthetic-finite-media-v1'
PAGE5_PROFILE = 'synthetic-finite-media-page5-v1'
PLEX_PROFILE = 'synthetic-finite-plex-v1'
PROFILES = (PROFILE, PAGE5_PROFILE, PLEX_PROFILE)


def coverage(pages, source_pages):
    """Independent complete-catalog evidence; no rubric or model repair."""
    expected = {(kind, row['title']) for page in source_pages
                for kind, key in (('movie', 'movies'), ('show', 'series'))
                for row in page[key]}
    observed = set()
    for page in pages:
        if 'media' in page:
            observed.update((row['type'], row['title']) for row in page['media'])
        else:
            observed.update((kind, row['title'])
                for kind, key in (('movie', 'movies'), ('show', 'series'))
                for row in page[key])
    return dict(passed=observed == expected, expected_rows=len(expected),
                observed_rows=len(observed), missing_rows=len(expected-observed),
                unexpected_rows=len(observed-expected))


def respond(call, source_pages, *, profile=PROFILE):
    if profile not in PROFILES:
        raise ValueError('unknown finite media profile')
    library = (profile == PLEX_PROFILE
               and call.get('name') == 'plugin__plex__plex_list_library')
    if not library and call.get('name') != 'plugin__plex__plex_list_library_media':
        raise ValueError('finite media fixture only supports the media endpoint')
    args = call.get('arguments')
    if not isinstance(args, dict):
        raise ValueError('media arguments must be an object')
    kind = args.get('mediaType') or 'all'
    if kind not in ('all', 'movie', 'show'):
        raise ValueError('invalid media type')
    def integer(name, default, minimum):
        value = args.get(name)
        value = default if value is None else value
        if type(value) not in (int, float) or not float(value).is_integer() or value < minimum:
            raise ValueError('invalid ' + name)
        return int(value)
    cap = 3 if library else 5 if profile in (PAGE5_PROFILE, PLEX_PROFILE) else 500
    limit = min(cap, integer('limit', 100, 1))
    offset = integer('offset', 0, 0)
    # Fail visibly if an unsupported query needs semantics this finite catalog
    # cannot provide, rather than silently claiming the filter was applied.
    for name in ('query', 'sectionId', 'sectionName', 'year', 'minYear', 'maxYear'):
        if args.get(name) is not None:
            raise ValueError('unsupported finite-catalog filter: ' + name)
    if library:
        # Separate movie/series streams share the caller's offset, matching
        # the library endpoint's two has-more witnesses. This explicit fixture
        # has a three-record cap PER TYPE, not the media endpoint's combined
        # five-record cap. No filtering, call/argument repair or shared cursor.
        result = dict(limit=limit, offset=offset, filtersApplied=False,
            notice='Synthetic raw catalog. Rating, root-location and library-section filters are NOT applied; verify every returned record against the user criteria.')
        for media_type, key, singular in (('movie', 'movies', 'movie'),
                                         ('show', 'series', 'series')):
            rows = [copy.deepcopy(row) for page in source_pages for row in page[key]]
            rows = sorted(rows, key=lambda row: row['title']) if kind in ('all', media_type) else []
            selected = rows[offset:offset + limit]
            result[key] = selected
            result[singular + 'Total'] = len(rows)
            result[singular + 'Returned'] = len(selected)
            result[singular + 'HasMore'] = offset + len(selected) < len(rows)
        return result
    rows = []
    for page in source_pages:
        for media_type, key in (('movie', 'movies'), ('show', 'series')):
            if kind not in ('all', media_type):
                continue
            for source in page[key]:
                row = copy.deepcopy(source)
                row['type'] = media_type
                row['sectionName'] = row.pop('plexLibrarySectionName')
                rows.append(row)
    # Stable order does not depend on previous calls or which query ran first.
    rows.sort(key=lambda row: (row['title'], row['type']))
    selected = rows[offset:offset + limit]
    return dict(total=len(rows), returned=len(selected), limit=limit, offset=offset,
        hasMore=offset + len(selected) < len(rows), media=selected,
        filtersApplied=False,
        notice='Synthetic raw catalog. Rating and root-location filters are NOT applied; verify every returned record against the user criteria.')
