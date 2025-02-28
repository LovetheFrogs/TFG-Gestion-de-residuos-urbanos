"""Definition of custom exceptions used in the codebase"""

class NodeNotFound(Exception):
    """Used when searching for an object that is not found in the structure
    
    Args:
        message (optional): Description of the error, default is 
            ``The node searched for was not found in the structure. Index 
            searched:``.
        id: Id of the not found node.
        *args: Variable length argument list.
    """
    def __init__(
        self, 
        id: int, 
        message: str = "The node searched for was not found in the structure."
        " Index searched:",
        *args
    ):
        self.message = message
        self.id = id
        super(NodeNotFound, self).__init__(f"{self.message} {self.id}", *args)


class DuplicateNode(Exception):
    """Used if a Node is already in a graph.
    
    Args:
        message (optional): Description of the error, default is 
            `The node is already in the Graph`.
        *args: Variable length argument list.
    """
    def __init__(self, message: str = "The node is already in the Graph", 
        *args
    ):
        self.message = message
        super(DuplicateNode, self).__init__(message, *args)


class DuplicateEdge(Exception):
    """Used if an Edge is already in a graph.

    Args:
        message (optional): Description of the error, 
            default is `The edge is already in the Graph`.
        *args: Variable length argument list.
    """
    def __init__(self, message: str = "The edge is already in the Graph", 
        *args
    ):
        self.message = message
        super(DuplicateEdge, self).__init__(message, *args)


class EdgeNotFound(Exception):
    """Used if an Edge is not found in the graph.
    
    Args:
        path: Source and destination nodes of the unfound edge.
        message (optional): Description of the error, 
            default is `The edge was not found in the structure.`
        *args: Variable length argument list.
    """
    def __init__(
        self,
        path: str,
        message: str = "The edge was not found in the structure.",
        *args
    ):
        self.message = message
        super(EdgeNotFound, self).__init__(f"{message} Edge {path}", *args)


class NoCenterDefined(Exception):
    """Used if a graph does not have any central node.
    
    Args:
        message (optional): Description of the error, 
            default is `A node has not been set to be the center of the graph.`
        *args: Variable length argument list.
    """
    def __init__(
        self,
        message: str = "A node has not been set to be the center of the graph.",
        *args
    ):
        self.message = message
        super(NoCenterDefined, self).__init__(message, *args)


class EmptyGraph(Exception):
    """Used when a graph does not have any nodes and/or edges in it.
    
    Args:
        message (optional): Description of the error, 
            default is `The graph does not have any edges or nodes in it.`
        *args: Variable length argument list.
    """
    def __init__(
        self, 
        message: str = "The graph does not have any edges or nodes in it.", 
        *args
    ):
        self.message = message
        super(EmptyGraph, self).__init__(message, *args)
