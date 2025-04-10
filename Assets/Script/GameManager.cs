using System.Diagnostics;
using UnityEngine;
using UnityEngine.SceneManagement;
using WebSocketClient;

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

            InitialiseNetworkControl();
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

        // this function calls when the current game state is updated to "Fight"
        //  and allows for the control of the P2 character via the network.
        void InitialiseNetworkControl()
        {

            // establishing client-server connection
            try{
                Process pythonServer = Process.Start(@"NeuralNetwork\server.py");
                _ = FootsiesClient.Main();
                
                UnityEngine.Debug.Log("Initialised server-client connection");
            }

            catch{
                UnityEngine.Debug.Log("Failed to establish server-client connection.");
            }

        }

        // this function relenquishes network control of the P2 character
        void RelenquishNetworkControl()
        {

            UnityEngine.Debug.Log("Relenquished Network Control");

        }
    
    }

}