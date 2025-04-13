import runpod

def handler(job):
    """
    A simple handler that returns hello world for any input.
    """
    return "hello world"

if __name__ == '__main__':
    runpod.serverless.start({
        'handler': handler
    })