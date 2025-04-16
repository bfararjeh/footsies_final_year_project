using System;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Footsies
{
    public class GameManager : Singleton<GameManager>
    {
        public enum SceneIndex
        {
            Title = 1,
            Battle = 2,
        }

        public AudioClip menuSelectAudioClip;

        public SceneIndex currentScene { get; private set; }
        public bool isVsCPU { get; private set; }

        private void Awake()
        {
            DontDestroyOnLoad(this.gameObject);

            Application.targetFrameRate = 60;
        }

        private void Start()
        {
            LoadTitleScene();
            InitialiseNetworkControl();
        }

        private void Update()
        {
            if(currentScene == SceneIndex.Battle)
            {
                if(Input.GetButtonDown("Cancel"))
                {
                    LoadTitleScene();
                }
            }
        }

        public void LoadTitleScene()
        {
            SceneManager.LoadScene((int)SceneIndex.Title);
            currentScene = SceneIndex.Title;

        }

        [System.Obsolete]
        public void LoadVsPlayerScene()
        {
            isVsCPU = false;
            LoadBattleScene();
        }

        [System.Obsolete]
        public void LoadVsCPUScene()
        {
            isVsCPU = true;
            LoadBattleScene();
        }

        [System.Obsolete]
        private void LoadBattleScene()
        {
            SceneManager.LoadScene((int)SceneIndex.Battle);
            currentScene = SceneIndex.Battle;

            if(menuSelectAudioClip != null)
            {
                SoundManager.Instance.playSE(menuSelectAudioClip);
            }
        }

        /*
        the method that launches the Python server and Footsies client
        called at program start
        */
        void InitialiseNetworkControl()
        {
            try
            {
                Process pythonServer = new();
                pythonServer.StartInfo.FileName = @"NeuralNetwork\server.py";
                pythonServer.Start();

                _ = WebSocketClient.FootsiesClient.Main();

                UnityEngine.Debug.Log("Client and Server Launched");
            }

            catch (Exception ex)
            {
                UnityEngine.Debug.Log("Failed to establish server-client connection.");
                UnityEngine.Debug.Log($"{ex.Message}");
            }

        }
    }

}