import config
import ipdb
from dataccess import query
import numpy as np
import time

def test_DataSet():
    # Edge case test
    assert query.DataSet([])

def test_query_generic():
    q = query.construct_query('label', '.*')
    runs = query.query_generic(q.attribute, q.function)
    target =\
        set([40, 379, 436, 905, 906, 907, 530, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552,
        553, 554, 555, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599,
        600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615,
        616, 617, 618, 619, 620, 621])
    assert runs == target

def test_parse_list_of_strings_to_query():
    assert query.parse_list_of_strings_to_query(['transmission', '1','2', 'runs', '2', 'bar']) == [('transmission', 1.0, 2.0), ('runs', '2', 'bar')]
    assert query.parse_list_of_strings_to_query(['transmission', '1', 'runs', '2', '3']) == [('transmission', 1.0), ('runs', 2.0, 3.0)]

def test_main():
    slist = "runs 530 535".split()
    dataset1 = query.main(slist)
    time.sleep(1)
    frame1, edd1 = dataset1.evaluate('si')
    f = config.every_other_filter
    dataset2 = query.main(slist, event_filter = f, event_filter_detid = 'si')
    time.sleep(1)
    frame2, edd2 = dataset2.evaluate('si')
    assert not np.all(frame1 == frame2)
    return edd1, edd2

def test_main_2():
    first = query.main('material Fe3O4 runs 876 961 transmission 1'.split())
    second = query.main('material Fe3O4HEF runs 876 961 transmission 1'.split())
    third = query.main('material Fe3O4 runs 876 961 transmission 1'.split())
    assert first.runs == third.runs
    
