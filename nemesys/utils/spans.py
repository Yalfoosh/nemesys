def get_spans_from_indices(indices, is_sorted=False):
    if not is_sorted:
        indices = sorted(indices)

    indices = list(indices)
    spans = list()

    current_span = [indices[0], indices[0] + 1]

    for index in indices[1:]:
        if index != current_span[1]:
            spans.append(tuple(current_span))
            current_span = [index, index + 1]
        else:
            current_span[1] += 1

    spans.append(tuple(current_span))

    return spans
