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


class EdgeNotFound(Exception):
    """ Used if an Edge is not found in the graph.
    
    Parameters
    ----------
    path : str
        Source and destination nodes of the unfound edge.
    message : str, optional
        Description of the error, default is `The edge was not found in the structure.`
    """
    def __init__(self, path, message = "The edge for was not found in the structure.", *args):
        self.message = message
        super(EdgeNotFound, self).__init__(f"{message} Edge {path}", *args)


class NoCenterDefined(Exception):
    """ Used if a graph does not have any central node.
    
    Parameters
    ----------
    message : str, optional
        Description of the error, default is `A node has not been set to be the center of the graph.`
    """
    def __init__(self, message = "A node has not been set to be the center of the graph.", *args):
        self.message = message
        super(NoCenterDefined, self).__init__(message, *args)


class EmptyGraph(Exception):
    """ Used when a graph does not have any nodes and/or edges in it.
    
    Parameters
    ----------
    message : str, optional
        Description of the error, default is `The graph does not have any edges or nodes in it.`
    """
    def __init__(self, message = "The graph does not have any edges or nodes in it.", *args):
        self.message = message
        super(EmptyGraph, self).__init__(message, *args)
