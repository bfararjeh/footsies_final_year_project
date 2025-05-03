from gameAI import FootsiesPredictor
import tensorflow as tf
import asyncio, websockets, time, sys

shutdownEvent = asyncio.Event()

async def messageHandler(websocket):
    '''
    This function handles all messages sent by the client in an asynchronous
        way
    The server is opened on port 8677 and this function is run asynchronously
        to receieve and then process all messages.
    Argument type: websocket
    '''

    try:
        # this block is where message recieving and processing happens
        async for message in websocket:

            footsiesAI.prepareData(message)

            start = time.perf_counter()
            output = footsiesAI.predict()
            if (time.perf_counter() - start)*1000 > 0.01:
                print(f"Predict: {(time.perf_counter() - start)*1000:.2f}ms")

            await websocket.send(str(output))
    
    # exception handling
    except websockets.ConnectionClosed:
        print("Client disconnected.")
        shutdownEvent.set()

    except Exception as e:
        print(f"Error: {e}.")
        shutdownEvent.set()

    finally:
        print("Connection closed.")
        shutdownEvent.set()

# start the WebSocket server
async def control_server():

    server = await websockets.serve(messageHandler, 
                                    "localhost", 
                                    8677,
                                    ping_interval=5,
                                    ping_timeout=10)
    
    print("WebSocket server started on ws://localhost:8677")
    await shutdownEvent.wait()
    print("\nPlease press enter to shutdown the server.")
    input()

# run the server
if __name__ == "__main__":

    try:

        # creates instance of the predictor class
        footsiesAI = FootsiesPredictor(modelPath="FootsiesNeuralNetwork.keras",
                                       sequenceLength=30,
                                       features=16,
                                       predictIntervals=5)
        
        asyncio.run(control_server())

    except Exception as e:
        print(f"Failed initialise server and/or network: {e}")
        sys.exit(1)