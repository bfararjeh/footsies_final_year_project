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
            
            using (ClientWebSocket client = new())
            {
                Uri serverUri = new("ws://localhost:8677");
                
                try
                {
                    // Connect to the server
                    await client.ConnectAsync(serverUri, CancellationToken.None);
                    UnityEngine.Debug.Log("Connected to the server");
                    
                    // Start a task to receive messages
                    var receiveTask = ReceiveMessagesAsync(client);
                    
                    // This block is where message sending happens
                    while (client.State == WebSocketState.Open)
                    {
                        Console.Write("Enter message (or 'exit' to quit): ");
                        string message = Console.ReadLine();
                        
                        if (message.ToLower() == "exit")
                            break;
                        
                        // Send the message
                        byte[] messageBytes = Encoding.UTF8.GetBytes(message);
                        await client.SendAsync(new ArraySegment<byte>(messageBytes), 
                                              WebSocketMessageType.Text, 
                                              true, 
                                              CancellationToken.None);
                    }
                    
                    // Close the connection gracefully
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
            Console.ReadKey();
        }
        
        public static async Task ReceiveMessagesAsync(ClientWebSocket client)
        {
            byte[] buffer = new byte[1024];
            
            while (client.State == WebSocketState.Open)
            {
                try
                {
                    System.Threading.Thread.Sleep(50);
                    int retryAttempts = 0;
                    var result = await client.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
                    while (true){

                        retryAttempts ++;

                        if (result == null && retryAttempts < 3){
                            result = await client.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
                            retryAttempts ++;
                        }
                        else{
                            break;
                        }
                    }

                    if (retryAttempts == 3){
                        UnityEngine.Debug.Log("Failed to secure connection (Retry Limit Reached)");
                        break;
                    }

                    else if (result.MessageType == WebSocketMessageType.Text)
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