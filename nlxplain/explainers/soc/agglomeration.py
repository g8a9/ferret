import numpy as np

# Magical code from ACD

def collapse_tree(lists):
    num_iters = len(lists['comps_list'])
    num_words = len(lists['comps_list'][0])

    # need to update comp_scores_list, comps_list
    comps_list = [np.zeros(num_words, dtype=np.int) for i in range(num_iters)]
    comp_scores_list = [{0: 0} for i in range(num_iters)]
    comp_levels_list = [{0: 0} for i in range(num_iters)]  # use this to determine what level to put things at

    # initialize first level
    comps_list[0] = np.arange(num_words)
    comp_levels_list[0] = {i: 0 for i in range(num_words)}

    # iterate over levels
    for i in range(1, num_iters):
        comps = lists['comps_list'][i]
        comps_old = lists['comps_list'][i - 1]
        comp_scores = lists['comp_scores_list'][i]

        for comp_num in range(1, np.max(comps) + 1):
            comp = comps == comp_num
            comp_size = np.sum(comp)
            if comp_size == 1:
                comp_levels_list[i][comp_num] = 0  # set level to 0
            else:
                # check for matches
                matches = np.unique(comps_old[comp])
                num_matches = matches.size

                # if 0 matches, level is 1
                if num_matches == 0:
                    level = 1
                    comp_levels_list[i][comp_num] = level  # set level to level 1

                # if 1 match, maintain level
                elif num_matches == 1:
                    level = comp_levels_list[i - 1][matches[0]]


                # if >1 match, take highest level + 1
                else:
                    level = np.max([comp_levels_list[i - 1][match] for match in matches]) + 1

                comp_levels_list[i][comp_num] = level
                new_comp_num = int(np.max(comps_list[level]) + 1)
                comps_list[level][comp] = new_comp_num  # update comp
                comp_scores_list[level][new_comp_num] = comp_scores[comp_num]  # update comp score

    # remove unnecessary iters
    num_iters = 0
    while np.sum(comps_list[num_iters] > 0) and num_iters < len(comps_list):
        num_iters += 1

    # populate lists
    lists['comps_list'] = comps_list[:num_iters]
    lists['comp_scores_list'] = comp_scores_list[:num_iters]
    return lists

# threshold scores at a specific percentile
def threshold_scores(scores, percentile_include, absolute):
    # pick based on abs value?
    if absolute:
        scores = np.absolute(scores)
    if type(scores) is list:
        scores = np.array(scores)

    # last 5 always pick 2
    num_left = scores.size - np.sum(np.isnan(scores))
    if num_left <= 5:
        #if num_left == 5:
        #    percentile_include = 59
        #elif num_left == 4:
        #    percentile_include = 49
        #if num_left == 3:
        #    percentile_include = 59
        #if num_left == 2:
        #    percentile_include = 49
        #elif num_left == 1:
        #    percentile_include = 0
        pass
    thresh = np.nanpercentile(scores, percentile_include)
    mask = scores >= thresh
    return mask

# pytorch needs to return each input as a column
# return batch_size x L tensor
def gen_tiles(text, fill=0,
              method='occlusion', prev_text=None, sweep_dim=1):
    L = text.shape[0]
    texts = np.zeros((L - sweep_dim + 1, L), dtype=np.int)
    for start in range(L - sweep_dim + 1):
        end = start + sweep_dim
        if method == 'occlusion':
            text_new = np.copy(text).flatten()
            text_new[start:end] = fill
        elif method == 'build_up' or method == 'cd':
            text_new = np.zeros(L)
            text_new[start:end] = text[start:end]
        texts[start] = np.copy(text_new)
    return texts


# return tile representing component
def gen_tile_from_comp(text_orig, comp_tile, method, fill=0):
    if method == 'occlusion':
        tile_new = np.copy(text_orig).flatten()
        tile_new[comp_tile] = fill
    elif method == 'build_up' or method == 'cd':
        tile_new = np.zeros(text_orig.shape)
        tile_new[comp_tile] = text_orig[comp_tile]
    return tile_new


# generate tiles around component
def gen_tiles_around_baseline(text_orig, comp_tile, method='build_up', sweep_dim=1, fill=0):
    L = text_orig.shape[0]
    left = 0
    right = L - 1
    while not comp_tile[left]:
        left += 1
    while not comp_tile[right]:
        right -= 1
    left = max(0, left - sweep_dim)
    right = min(L - 1, right + sweep_dim)
    tiles = []
    for x in [left, right]:
        if method == 'occlusion':
            tile_new = np.copy(text_orig).flatten()
            tile_new[comp_tile] = fill
            tile_new[x] = fill
        elif method == 'build_up' or method == 'cd':
            tile_new = np.zeros(text_orig.shape)
            tile_new[comp_tile] = text_orig[comp_tile]
            tile_new[x] = text_orig[x]
        tiles.append(tile_new)
    return np.array(tiles), [left, right]

def tiles_to_cd(tiles):
    starts, stops = [], []
    #tiles = batch.text.data.cpu().numpy()
    L = tiles.shape[0]
    for c in range(tiles.shape[1]):
        text = tiles[:, c]
        start = 0
        stop = L - 1
        while text[start] == 0:
            start += 1
        while text[stop] == 0:
            stop -= 1
        starts.append(start)
        stops.append(stop)
    return starts, stops

def lists_to_tabs(lists, num_words):

    num_iters = len(lists['comps_list'])
    data = np.empty(shape=(num_iters, num_words))
    data[:] = np.nan
    data[0, :] = lists['scores_list'][0]
    for i in range(1, num_iters):
        comps = lists['comps_list'][i]
        comp_scores_list = lists['comp_scores_list'][i]

        for comp_num in range(1, np.max(comps) + 1):
            idxs = comps == comp_num
            data[i][idxs] = comp_scores_list[comp_num]
    data[np.isnan(data)] = 0  # np.nanmin(data) - 0.001
    return data

