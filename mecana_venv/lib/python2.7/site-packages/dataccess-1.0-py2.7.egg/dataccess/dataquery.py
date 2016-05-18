from collections import namedtuple
import numpy as np

QueryResultBase = namedtuple('QueryResultBase', ['mean_frame', 'event_data', 'nevents'])
class QueryResult(QueryResultBase):
    def __new__(self, mean_frame, event_data, nevents):
#        assert isinstance(mean_frame, np.ndarray)
#        assert isinstance(event_data, dict) 
#        assert isinstance(nevents, int)
        return QueryResultBase.__new__(QueryResult, mean_frame, event_data, nevents)
    def __add__(self, other):
        """
        Add two QueryResult instances.
        """
        nevents = self.nevents + other.nevents
        event_data = utils.merge_dicts(self.event_data, other.event_data)
        mean_frame = (self.nevents * self.mean_frame + other.nevents * other.mean_frame) / float(self.nevents + other.nevents)
        return QueryResult(mean_frame, event_data, nevents)
    
#def serve_dataset_query(detid, runs, event_filter, event_data_getter):
#    """
#    Arguments:
#    detid : str
#        detector ID
#    runs : list of ints
#        run numbers
#    event_filter : EventFilter
#        specifies events to accept.
#    event_data_getter : function
#        Function with which to compute event-by-event derived data.
#    Returns a DatasetQueryResult instance.
#    """
    
