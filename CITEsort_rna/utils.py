import time


class Utils(object):
    @classmethod
    def timer(
        cls,
        func
    ):
        """
        Timer decorator for methods.

        Args:
            func (_type_): _description_
        """
        def timer_func(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            msg = f'Function {func.__name__!r} executed in {(end_time-start_time):.2f}s'
            print('-' * len(msg))
            print(msg)
            print('-' * len(msg))
            return result
        return timer_func
