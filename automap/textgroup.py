




def connect_text(data, ythresh=6, xthresh=6):
    
    def merge_textgroups(newdata):
        for i in range(len(newdata)):
            group = newdata[i]
            dct = {'text': ' '.join([r['text'] for r in group]),
                   'text_clean': ' '.join([r['text_clean'] for r in group]),
                   #'numeric': min([r['numeric'] for r in group]),
                   #'uppercase': min([r['uppercase'] for r in group]),
                   'conf': sum([r['conf'] for r in group]) / float(len(group)),
                   'left': min([r['left'] for r in group]),
                   'top': min([r['top'] for r in group]),
                   'fontheight': max([r['fontheight'] for r in group]),
                   #'function': group[0]['function'],
                   'color': group[0]['color'],
                   }
            dct['width'] = max([r['left']+r['width'] for r in group]) - dct['left']
            dct['height'] = max([r['top']+r['height'] for r in group]) - dct['top']
            #print len(group),dct
            newdata[i] = dct
        return newdata
    
    # connect texts horizontally
    candidates = sorted(data, key=lambda r: r['left'])
    newdata = []
    while candidates:
        r = candidates.pop(0)
        # find all whose top and bottom are within threshold
        totheright = []
        for r2 in candidates:
            if r2 == r: continue
            # height difference can't be more than x2
            if (max(r['height'],r2['height']) / float(min(r['height'],r2['height']))) > 2: 
                continue
            # top or bottom within threshold
            if (abs(r['top'] - r2['top']) < ythresh) or (abs((r['top']+r['height']) - (r2['top']+r2['height'])) < ythresh): 
                totheright.append(r2)
        # group those within height x distance
        group = [r]
        right = r['left'] + r['width']
        while totheright:
            nxt = totheright.pop(0)
            if (right + r['height']) > nxt['left']:
                # within distance, add to group
                right = nxt['left'] + nxt['width']
                candidates.pop(candidates.index(nxt)) # remove as candidate for others
                group.append(nxt)
            else:
                # not within distance, break loop
                break
        newdata.append(group)

    # merge groups
    newdata = merge_textgroups(newdata)

    # do same vertically (center aligned only)
    candidates = sorted(newdata, key=lambda r: r['top'])
    newdata = []
    while candidates:
        r = candidates.pop(0)
        # find all whose midpoints are within threshold
        below = []
        for r2 in candidates:
            if r2 == r: continue
            # height difference can't be more than x2
            if (max(r['height'],r2['height']) / float(min(r['height'],r2['height']))) > 2: 
                continue
            # midpoints within threshold
            mid1 = r['left'] + (r['width'] / 2.0)
            mid2 = r2['left'] + (r2['width'] / 2.0)
            if abs(mid1 - mid2) < xthresh:
                below.append(r2)
        # group those within height y distance
        group = [r]
        bottom = r['top'] + r['height']
        while below:
            nxt = below.pop(0)
            #print '---'
            #print bottom + r['height'], nxt['top']
            #print r
            #print nxt
            if (bottom + r['height']) > nxt['top']:
                # within distance, add to group
                bottom = nxt['top'] + nxt['height']
                candidates.pop(candidates.index(nxt)) # remove as candidate for others
                group.append(nxt)
            else:
                # not within distance, break loop
                break
        newdata.append(group)

    # merge groups
    newdata = merge_textgroups(newdata)

##    # merge text groups vertically (center aligned only)
##    candidates = sorted(newdata, key=lambda gr: min([r['top'] for r in gr]))
##    newdata2 = []
##    while candidates:
##        gr = candidates.pop(0)
##        below = []
##        # find all exactly below, ie whose midpoint is within threshold
##        for gr2 in candidates:
##            if gr2 == gr: continue
##            mid1 = (gr[0]['left'] + (gr[-1]['left'] + gr[-1]['width']) / 2.0)
##            mid2 = (gr2[0]['left'] + (gr2[-1]['left'] + gr2[-1]['width']) / 2.0)
##            if abs(mid1 - mid2) < xthresh:
##                # same midpoint
##                below.append(gr2)
##        # group those witihin height y distance
##        bottom = max([r['top']+r['height'] for r in gr])
##        grheight = bottom - min([r['top'] for r in gr])
##        while below:
##            nxtgr = below.pop(0)
##            if (bottom + grheight) > min([r['top'] for r in nxtgr]):
##                # within height distance, add to group
##                bottom = max([r['top']+r['height'] for r in nxtgr])
##                candidates.pop(candidates.index(nxtgr))
##                gr.extend(nxtgr)
##            else:
##                # not within distance, break loop
##                break
##        newdata2.append(gr)   

    return newdata



