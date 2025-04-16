import asyncio
import websockets

# message handler for connected clients
async def message_handler(websocket):

    try:
        # this block is where message recieving and processing happens
        async for message in websocket:
            print(f"Received message: {message}")
            await websocket.send(f"Server received: {message}")
    
    # exception handling
    except Exception as e:
        print(f"Error: {e}.\nConnection closed. Press enter to terminate server.")
        input()
        control_server(0)

    finally:
        print("Connection closed. Press enter to terminate server.")
        input()
        control_server(0)


# start the WebSocket server
async def control_server(controlCommand):

    if controlCommand == 1:
        server = await websockets.serve(message_handler, "localhost", 8677)
        print("WebSocket server started on ws://localhost:8677")
        await server.wait_closed()

    else:
        server.close()


# run the server
if __name__ == "__main__":

    try:
        asyncio.run(control_server(1))

    except Exception as e:
        print(f"Error: {e}")
        input()