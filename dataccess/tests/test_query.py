import config
from dataccess import query

def test_main():
    # Edge case test
    assert query.DataSet([])

def test_query_generic():
    q = query.construct_query('label', '.*')
    runs = query.query_generic(q.attribute, q.function)
    target =\
        set([905, 906, 907, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552,
        553, 554, 555, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599,
        600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615,
        616, 617, 618, 619, 620, 621])
    assert runs == target

