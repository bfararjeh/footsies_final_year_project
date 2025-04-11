using System;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace WebSocketClient
{

    class FootsiesClient
    {
        public static async Task Main()
        {
            UnityEngine.Debug.Log("WebSocket Client Starting...");
            
            // sets up a web server locally via port 8677
            using (ClientWebSocket client = new())
            {
                Uri serverUri = new("ws://localhost:8677");
                
                try
                {
                    // connect to the server
                    await client.ConnectAsync(serverUri, CancellationToken.None);
                    UnityEngine.Debug.Log("Connected to the server");
                    
                    // start a task to receive messages
                    var receiveTask = ReceiveMessagesAsync(client);
                    
                    // this block is where message sending happens
                    // until proper code is written, a test message is sent
                    //  repeatedly
                    while (client.State == WebSocketState.Open)
                    {
                        UnityEngine.Debug.Log("Sending test message...");
                        string message = "Test message";
                        
                        // the message is encoded into UTF8 before being sent
                        byte[] messageBytes = Encoding.UTF8.GetBytes(message);
                        await client.SendAsync(new ArraySegment<byte>(messageBytes), 
                                              WebSocketMessageType.Text, 
                                              true, 
                                              CancellationToken.None);
                    }
                    
                    // close the connection gracefully
                    // this code is called when a server close message is
                    //  recieved
                    await client.CloseAsync(WebSocketCloseStatus.NormalClosure, 
                                          "Client closing", 
                                          CancellationToken.None);
                }
                catch (Exception ex)
                {
                    UnityEngine.Debug.Log($"Error: {ex.Message}");
                }
            }
            
            UnityEngine.Debug.Log("Client shut down");
        }
        
        public static async Task ReceiveMessagesAsync(ClientWebSocket client)
        {
            // creates a buffer to recieve the message into
            byte[] buffer = new byte[1024];
            
            while (client.State == WebSocketState.Open)
            {
                try
                {
                    // recieves the message from the server
                    var result = await client.ReceiveAsync(
                        new ArraySegment<byte>(buffer), CancellationToken.None);

                    // logs the recieved message, or if the recieved message is
                    //  a close call, closes the connection
                    if (result.MessageType == WebSocketMessageType.Text)
                    {
                        string message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                        UnityEngine.Debug.Log($"Received: {message}");
                    }
                    else if (result.MessageType == WebSocketMessageType.Close)
                    {
                        await client.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, 
                                                   "Server closed connection", 
                                                   CancellationToken.None);
                        break;
                    }
                }
                catch (Exception ex)
                {
                    UnityEngine.Debug.Log($"Error receiving: {ex.Message}");
                    break;
                }
            }
        }
    
    }
}