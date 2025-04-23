"""Module containing utility functions."""

import time

# Internal stats for smoothing ETA calculations
__pb_stats = {}


def printProgressBar(iteration: int,
                     total: int,
                     prefix: str = '',
                     suffix: str = '',
                     decimals: int = 1,
                     length: int = 100,
                     fill: str = '█',
                     printEnd: str = "\r",
                     show_eta: bool = False):
    """Call in a loop to create terminal progress bar with optional smoothed countdown ETA.

    Args:
        iteration: Current iteration.
        total: Total iterations.
        prefix (optional): Prefix string. Defaults to ''.
        suffix (optional): Suffix string. Defaults to ''.
        decimals (optional): Positive number of decimals in percent complete. Defaults to 1.
        length (optional): Character length of bar. Defaults to 100.
        fill (optional): Bar fill character. Defaults to '█'.
        printEnd (optional): End character (e.g. "\r", "\n"). Defaults to "\r".
        show_eta (optional): Whether to show a smoothed ETA countdown. Defaults to False.
    """
    now = time.time()
    eta_str = ''

    if show_eta:
        # Initialize stats on first call
        if iteration == 0:
            __pb_stats['default'] = {'last_time': now, 'ema': None}
        else:
            stats = __pb_stats.get('default')
            dt = now - stats['last_time']
            # smoothing factor: smaller = smoother
            alpha = 0.2
            if stats['ema'] is None:
                stats['ema'] = dt
            else:
                stats['ema'] = alpha * dt + (1 - alpha) * stats['ema']
            stats['last_time'] = now

            # estimate remaining seconds and format as HH:MM:SS
            remaining = total - iteration
            eta_seconds = int(stats['ema'] * remaining)
            hrs, rem = divmod(eta_seconds, 3600)
            mins, secs = divmod(rem, 60)
            eta_str = f" | ETA: {hrs:02}:{mins:02}:{secs:02}"

    # build percentage and bar
    percent = (f"{{0:.{decimals}f}}".format(100 * (iteration / float(total)))
               if total else "100.0")
    filledLength = int(length * iteration // total) if total else length
    bar = fill * filledLength + '-' * (length - filledLength)

    # print to console
    print(f'\r{prefix} |{bar}| {percent}% {suffix}{eta_str}', end=printEnd)

    # Print New Line on Complete
    if iteration == total:
        print()
