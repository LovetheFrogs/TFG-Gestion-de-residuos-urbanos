"""Module containing utility functions."""

def printProgressBar(
        iteration: int, 
        total: int, 
        prefix: str = '', 
        suffix: str = '', 
        decimals: int = 1,
        length: int = 100,
        fill: str = '█',
        printEnd: str = "\r"
    ):
        """Call in a loop to create terminal progress bar
        
        Thanks to StackOverflow's Greenstick user for this function, extracted
        from an answer to a thread about progressbars in python.

        Args:
            iteration: Current iteration.
            total: Total iterations.
            prefix (optional): Prefix string. Defaults to ''.
            suffix (optional): Suffix string. Defaults to ''.
            decimals (optional): positive number of decimals in percent 
                complete. Defaults to 1.
            length (optional): character length of bar. Defaults to 100.
            fill (optional): bar fill character. Defaults to '█'.
            printEnd (optional): end character (e.g. "\r", "\r\n").
                Defaults to "\r".
        """
        percent = (
            "{0:." + str(decimals) + "f}"
        ).format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()
            