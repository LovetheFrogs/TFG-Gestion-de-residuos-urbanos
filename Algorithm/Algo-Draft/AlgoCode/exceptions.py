""" Definition of custom exceptions used in the codebase """

class NodeNotFound(Exception):
    """ Used when searching for an object that is not found in the structure
    
    Parameters
    ----------
    message : str, optional
        Description of the error, default is `The node of index {id} was not found in the structure`.
    id : int
        Id of the not found node.
    """
    def __init__(self, id, message = "The node searched for was not found in the structure. Index searched: ", *args):
        self.message = message
        self.id = id
        super(NodeNotFound, self).__init__(f"{message}{id}", *args)


class DuplicateNode(Exception):
    """ Used if a Node is already part of the graph.
    
    Parameters
    ----------
    message : str, opcional
        Description of the error, default is `The node is already in the Graph`.
    """
    def __init__(self, message = "The node is already in the Graph", *args):
        self.message = message
        super(DuplicateNode, self).__init__(message, *args)


class DuplicateEdge(Exception):
    """ Used if an Edge is already part of the graph.

    Parameters
    ----------
    message : str, optional
        Description of the error, default is `The edge is already in the Graph`.
    """
    def __init__(self, message = "The edge is already in the Graph", *args):
        self.message = message
        super(DuplicateEdge, self).__init__(message, *args)
