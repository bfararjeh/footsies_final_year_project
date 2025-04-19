using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Security;
using System.Threading;
using UnityEngine;
using System.Linq;

// disables unreadable code warning: just so i can comment out the output
//  training data function while i test
#pragma warning disable 0162

namespace Footsies
{
    /// <summary>
    /// Main update for battle engine
    /// Update player/ai input, fighter actions, hitbox/hurtbox collision, round start/end
    /// </summary>
    public class BattleCore : MonoBehaviour
    {
        public enum RoundStateType
        {
            Stop,
            Intro,
            Fight,
            KO,
            End,
        }

        [SerializeField]
        private float _battleAreaWidth = 10f;
        public float battleAreaWidth { get { return _battleAreaWidth; } }

        [SerializeField]
        private float _battleAreaMaxHeight = 2f;
        public float battleAreaMaxHeight { get { return _battleAreaMaxHeight; } }

        [SerializeField]
        private GameObject roundUI;

        [SerializeField]
        private List<FighterData> fighterDataList = new List<FighterData>();

        public bool debugP1Attack = false;
        public bool debugP2Attack = false;
        public bool debugP1Guard = false;
        public bool debugP2Guard = false;

        public bool debugPlayLastRoundInput = false;

        private float timer = 0;
        private uint maxRoundWon = 3;

        public Fighter fighter1 { get; private set; }
        public Fighter fighter2 { get; private set; }

        public uint fighter1RoundWon { get; private set; }
        public uint fighter2RoundWon { get; private set; }

        public List<Fighter> fighters { get { return _fighters; } }
        private List<Fighter> _fighters = new List<Fighter>();

        private float roundStartTime;
        private int frameCount;

        public RoundStateType roundState { get { return _roundState; } }
        private RoundStateType _roundState = RoundStateType.Stop;

        public System.Action<Fighter, Vector2, DamageResult> damageHandler;

        private Animator roundUIAnimator;

        private BattleAI battleAI = null;

        private static uint maxRecordingInputFrame = 60 * 60 * 5;
        private InputData[] recordingP1Input = new InputData[maxRecordingInputFrame];
        private InputData[] recordingP2Input = new InputData[maxRecordingInputFrame];
        private uint currentRecordingInputIndex = 0;

        private InputData[] lastRoundP1Input = new InputData[maxRecordingInputFrame];
        private InputData[] lastRoundP2Input = new InputData[maxRecordingInputFrame];
        private uint currentReplayingInputIndex = 0;
        private uint lastRoundMaxRecordingInput = 0;
        private bool isReplayingLastRoundInput = false;

        public bool isDebugPause { get; private set; }

        private float introStateTime = 3f;
        private float koStateTime = 2f;
        private float endStateTime = 3f;
        private float endStateSkippableTime = 1.5f;
        private string fightData;

        // variables to control data logging and outputting
        public string trainingData;
        public string newGameState;
        public int currentFrameCount;
        public bool dataLogged;
        public static int networkActive = 0;
        public static int networkInput = 0;

        // concurrentqueue ensures thread safety and synchronisation over queue
        // semaphore controls message sending without the need for a while loop
        //  which blocks the main game process
        public static readonly ConcurrentQueue<string> messageQueue = new();
        public static readonly SemaphoreSlim messageAvailable = new(0, int.MaxValue);
        

        void Awake()
        {

            // Setup dictionary from ScriptableObject data
            fighterDataList.ForEach((data) => data.setupDictionary());

            fighter1 = new Fighter();
            fighter2 = new Fighter();

            _fighters.Add(fighter1);
            _fighters.Add(fighter2);

            if(roundUI != null)
            {
                roundUIAnimator = roundUI.GetComponent<Animator>();
            }
        }

        [Obsolete]
        void FixedUpdate()
        {
            switch(_roundState)
            {
                case RoundStateType.Stop:

                    ChangeRoundState(RoundStateType.Intro);
                    // sets the frame count to 0 as well as marking data as
                    // having not been logged
                    // coulve used the preexisting frame count, but cba to
                    // figure it out lmao
                    // also empties the messagequeue
                    currentFrameCount = 0;
                    messageQueue.Clear();
                    dataLogged = false;

                    break;
                case RoundStateType.Intro:

                    UpdateIntroState();

                    timer -= Time.deltaTime;
                    if (timer <= 0f)
                    {
                        ChangeRoundState(RoundStateType.Fight);
                    }

                    if (debugPlayLastRoundInput
                        && !isReplayingLastRoundInput)
                    {
                        StartPlayLastRoundInput();
                    }

                    break;
                case RoundStateType.Fight:

                    if(CheckUpdateDebugPause())
                    {
                        break;
                    }

                    frameCount++;
                    
                    UpdateFightState();

                    var deadFighter = _fighters.Find((f) => f.isDead);
                    if(deadFighter != null)
                    {
                        ChangeRoundState(RoundStateType.KO);
                    }

                    break;
                case RoundStateType.KO:

                    UpdateKOState();
                    timer -= Time.deltaTime;
                    if (timer <= 0f)
                    {
                        ChangeRoundState(RoundStateType.End);
                    }

                    break;
                case RoundStateType.End:

                    UpdateEndState();
                    timer -= Time.deltaTime;
                    if (timer <= 0f
                        || (timer <= endStateSkippableTime && IsKOSkipButtonPressed()))
                    {
                        ChangeRoundState(RoundStateType.Stop);
                    }

                    break;
            }
        }

        [Obsolete]
        void ChangeRoundState(RoundStateType state)
        {
            _roundState = state;
            switch (_roundState)
            {
                case RoundStateType.Stop:

                    if(fighter1RoundWon >= maxRoundWon
                        || fighter2RoundWon >= maxRoundWon)
                    {
                        GameManager.Instance.LoadTitleScene();
                    }

                    break;
                case RoundStateType.Intro:

                    fighter1.SetupBattleStart(fighterDataList[0], new Vector2(-2f, 0f), true);
                    fighter2.SetupBattleStart(fighterDataList[0], new Vector2(2f, 0f), false);

                    timer = introStateTime;

                    roundUIAnimator.SetTrigger("RoundStart");

                    if (GameManager.Instance.isVsCPU)
                        battleAI = new BattleAI(this);

                    break;
                case RoundStateType.Fight:

                    roundStartTime = Time.fixedTime;
                    frameCount = -1;

                    currentRecordingInputIndex = 0;
                    
                    break;
                case RoundStateType.KO:

                    timer = koStateTime;

                    CopyLastRoundInput();

                    fighter1.ClearInput();
                    fighter2.ClearInput();

                    battleAI = null;

                    roundUIAnimator.SetTrigger("RoundEnd");

                    break;
                case RoundStateType.End:

                    timer = endStateTime;

                    var deadFighter = _fighters.FindAll((f) => f.isDead);
                    if (deadFighter.Count == 1)
                    {
                        if (deadFighter[0] == fighter1)
                        {
                            fighter2RoundWon++;
                            fighter2.RequestWinAction();
                        }
                        else if (deadFighter[0] == fighter2)
                        {
                            fighter1RoundWon++;
                            fighter1.RequestWinAction();
                        }
                    }

                    break;
            }
        }

        [Obsolete]
        void UpdateIntroState()
        {
            var p1Input = GetP1InputData();
            var p2Input = GetP2InputData();
            RecordInput(p1Input, p2Input);
            fighter1.UpdateInput(p1Input);
            fighter2.UpdateInput(p2Input);

            _fighters.ForEach((f) => f.IncrementActionFrame());

            _fighters.ForEach((f) => f.UpdateIntroAction());
            _fighters.ForEach((f) => f.UpdateMovement());
            _fighters.ForEach((f) => f.UpdateBoxes());

            UpdatePushCharacterVsCharacter();
            UpdatePushCharacterVsBackground();
        }

        [Obsolete]
        void UpdateFightState()
        {
            var p1Input = GetP1InputData();
            var p2Input = GetP2InputData();
            RecordInput(p1Input, p2Input);
            fighter1.UpdateInput(p1Input);
            fighter2.UpdateInput(p2Input);

            if (networkActive == 1)
            {
                p1Input.input = GetCurrentNetworkInput();
                fighter1.UpdateInput(p1Input);
            }


            _fighters.ForEach((f) => f.IncrementActionFrame());

            _fighters.ForEach((f) => f.UpdateActionRequest());
            _fighters.ForEach((f) => f.UpdateMovement());
            _fighters.ForEach((f) => f.UpdateBoxes());

            UpdatePushCharacterVsCharacter();
            UpdatePushCharacterVsBackground();
            UpdateHitboxHurtboxCollision();

            // increments the current frame (effectively a timecode)
            currentFrameCount++;

            try{

            // appends data to trainingData string
            newGameState = currentFrameCount + ": " +
            "P1_INFO:" + 
            "currentInput(" + p1Input.input +
            ")position" + fighter1.position +
            "velocity_x(" + fighter1.velocity_x +
            ")isDead(" + fighter1.isDead +
            ")vitalHealth(" + fighter1.vitalHealth +
            ")guardHealth(" + fighter1.guardHealth +
            ")currentActionID(" + fighter1.currentActionID +
            ")currentActionFrame(" + fighter1.currentActionFrame +
            ")currentActionFrameCount(" + fighter1.currentActionFrameCount +
            ")isAlwaysCancelable(" + fighter1.isAlwaysCancelable +
            ")currentActionHitCount(" + fighter1.currentActionHitCount +
            ")currentHitStunFrame(" + fighter1.currentHitStunFrame +
            ")isInHitStun(" + fighter1.isInHitStun +
            ")isAlwaysCancelable(" + fighter1.isAlwaysCancelable +
            ")" 
            +
            "P2_INFO:" + 
            "currentInput(" + p2Input.input +
            ")position" + fighter2.position +
            "velocity_x(" + fighter2.velocity_x +
            ")isDead(" + fighter2.isDead +
            ")vitalHealth(" + fighter2.vitalHealth +
            ")guardHealth(" + fighter2.guardHealth +
            ")currentActionID(" + fighter2.currentActionID +
            ")currentActionFrame(" + fighter2.currentActionFrame +
            ")currentActionFrameCount(" + fighter2.currentActionFrameCount +
            ")isAlwaysCancelable(" + fighter2.isAlwaysCancelable +
            ")currentActionHitCount(" + fighter2.currentActionHitCount +
            ")currentHitStunFrame(" + fighter2.currentHitStunFrame +
            ")isInHitStun(" + fighter2.isInHitStun +
            ")isAlwaysCancelable(" + fighter2.isAlwaysCancelable +
            ")\n";

            trainingData += newGameState;

            UpdateMessageQueue(newGameState);

            }

            catch
            {
                UnityEngine.Debug.Log("Error exporting game state.");
            }

        }

        void UpdateKOState()
        {

        }

        [Obsolete]
        void UpdateEndState()
        {
            _fighters.ForEach((f) => f.IncrementActionFrame());

            _fighters.ForEach((f) => f.UpdateActionRequest());
            _fighters.ForEach((f) => f.UpdateMovement());
            _fighters.ForEach((f) => f.UpdateBoxes());

            UpdatePushCharacterVsCharacter();
            UpdatePushCharacterVsBackground();

            // outputs the training data to the correct file
            // datalogged var ensures it only runs once
            if (dataLogged == false && fighter1.hasWon == true) {
                OutputTrainingData();
            }
        }

        [Obsolete]
        InputData GetP1InputData()
        {
            if(isReplayingLastRoundInput)
            {
                return lastRoundP1Input[currentReplayingInputIndex];
            }

            var time = Time.fixedTime - roundStartTime;

            InputData p1Input = new InputData();
            p1Input.input |= InputManager.Instance.GetButton(InputManager.Command.p1Left) ? (int)InputDefine.Left : 0;
            p1Input.input |= InputManager.Instance.GetButton(InputManager.Command.p1Right) ? (int)InputDefine.Right : 0;
            p1Input.input |= InputManager.Instance.GetButton(InputManager.Command.p1Attack) ? (int)InputDefine.Attack : 0;
            p1Input.time = time;

            if (debugP1Attack)
                p1Input.input |= (int)InputDefine.Attack;
            if (debugP1Guard)
                p1Input.input |= (int)InputDefine.Left;

            return p1Input;
        }

        [Obsolete]
        InputData GetP2InputData()
        {
            if (isReplayingLastRoundInput)
            {
                return lastRoundP2Input[currentReplayingInputIndex];
            }

            var time = Time.fixedTime - roundStartTime;

            InputData p2Input = new InputData();

            if (battleAI != null)
            {
                p2Input.input |= battleAI.getNextAIInput();
            }
            else
            {
                p2Input.input |= InputManager.Instance.GetButton(InputManager.Command.p2Left) ? (int)InputDefine.Left : 0;
                p2Input.input |= InputManager.Instance.GetButton(InputManager.Command.p2Right) ? (int)InputDefine.Right : 0;
                p2Input.input |= InputManager.Instance.GetButton(InputManager.Command.p2Attack) ? (int)InputDefine.Attack : 0;
            }

            p2Input.time = time;

            if (debugP2Attack)
                p2Input.input |= (int)InputDefine.Attack;
            if (debugP2Guard)
                p2Input.input |= (int)InputDefine.Right;

            return p2Input;
        }

        private int GetCurrentNetworkInput()
        {
            return networkInput;
        }  

        [Obsolete]
        private bool IsKOSkipButtonPressed()
        {
            if (InputManager.Instance.GetButton(InputManager.Command.p1Attack))
                return true;

            if (InputManager.Instance.GetButton(InputManager.Command.p2Attack))
                return true;

            return false;
        }
        
        void UpdatePushCharacterVsCharacter()
        {
            var rect1 = fighter1.pushbox.rect;
            var rect2 = fighter2.pushbox.rect;

            if (rect1.Overlaps(rect2))
            {
                if (fighter1.position.x < fighter2.position.x)
                {
                    fighter1.ApplyPositionChange((rect1.xMax - rect2.xMin) * -1 / 2, fighter1.position.y);
                    fighter2.ApplyPositionChange((rect1.xMax - rect2.xMin) * 1 / 2, fighter2.position.y);
                }
                else if (fighter1.position.x > fighter2.position.x)
                {
                    fighter1.ApplyPositionChange((rect2.xMax - rect1.xMin) * 1 / 2, fighter1.position.y);
                    fighter2.ApplyPositionChange((rect2.xMax - rect1.xMin) * -1 / 2, fighter1.position.y);
                }
            }
        }

        void UpdatePushCharacterVsBackground()
        {
            var stageMinX = battleAreaWidth * -1 / 2;
            var stageMaxX = battleAreaWidth / 2;

            _fighters.ForEach((f) =>
            {
                if (f.pushbox.xMin < stageMinX)
                {
                    f.ApplyPositionChange(stageMinX - f.pushbox.xMin, f.position.y);
                }
                else if (f.pushbox.xMax > stageMaxX)
                {
                    f.ApplyPositionChange(stageMaxX - f.pushbox.xMax, f.position.y);
                }
            });
        }

        [Obsolete]
        void UpdateHitboxHurtboxCollision()
        {
            foreach(var attacker in _fighters)
            {
                Vector2 damagePos = Vector2.zero;
                bool isHit = false;
                bool isProximity = false;
                int hitAttackID = 0;

                foreach (var damaged in _fighters)
                {
                    if (attacker == damaged)
                        continue;
                    
                    foreach (var hitbox in attacker.hitboxes)
                    {
                        // continue if attack already hit
                        if(!attacker.CanAttackHit(hitbox.attackID))
                        {
                            continue;
                        }

                        foreach (var hurtbox in damaged.hurtboxes)
                        {
                            if (hitbox.Overlaps(hurtbox))
                            {
                                if (hitbox.proximity)
                                {
                                    isProximity = true;
                                }
                                else
                                {
                                    isHit = true;
                                    hitAttackID = hitbox.attackID;
                                    float x1 = Mathf.Min(hitbox.xMax, hurtbox.xMax);
                                    float x2 = Mathf.Max(hitbox.xMin, hurtbox.xMin);
                                    float y1 = Mathf.Min(hitbox.yMax, hurtbox.yMax);
                                    float y2 = Mathf.Max(hitbox.yMin, hurtbox.yMin);
                                    damagePos.x = (x1 + x2) / 2;
                                    damagePos.y = (y1 + y2) / 2;
                                    break;
                                }
                                
                            }
                        }

                        if (isHit)
                            break;
                    }

                    if (isHit)
                    {
                        attacker.NotifyAttackHit(damaged, damagePos);
                        var damageResult = damaged.NotifyDamaged(attacker.getAttackData(hitAttackID), damagePos);

                        var hitStunFrame = attacker.GetHitStunFrame(damageResult, hitAttackID);
                        attacker.SetHitStun(hitStunFrame);
                        damaged.SetHitStun(hitStunFrame);
                        damaged.SetSpriteShakeFrame(hitStunFrame / 3);

                        damageHandler(damaged, damagePos, damageResult);
                    }
                    else if (isProximity)
                    {
                        damaged.NotifyInProximityGuardRange();
                    }
                }


            }
        }

        void RecordInput(InputData p1Input, InputData p2Input)
        {
            if (currentRecordingInputIndex >= maxRecordingInputFrame)
                return;

            recordingP1Input[currentRecordingInputIndex] = p1Input.ShallowCopy();
            recordingP2Input[currentRecordingInputIndex] = p2Input.ShallowCopy();
            currentRecordingInputIndex++;

            if (isReplayingLastRoundInput)
            {
                if (currentReplayingInputIndex < lastRoundMaxRecordingInput)
                    currentReplayingInputIndex++;
            }
        }

        void CopyLastRoundInput()
        {
            for(int i = 0; i < currentRecordingInputIndex; i++)
            {
                lastRoundP1Input[i] = recordingP1Input[i].ShallowCopy();
                lastRoundP2Input[i] = recordingP2Input[i].ShallowCopy();
            }
            lastRoundMaxRecordingInput = currentRecordingInputIndex;
            
            isReplayingLastRoundInput = false;
            currentReplayingInputIndex = 0;
        }

        void StartPlayLastRoundInput()
        {
            isReplayingLastRoundInput = true;
            currentReplayingInputIndex = 0;
        }

        bool CheckUpdateDebugPause()
        {
            if (Input.GetKeyDown(KeyCode.F1))
            {
                isDebugPause = !isDebugPause;
            }

            if (isDebugPause)
            {
                // press f2 during debug pause to 
                if (Input.GetKeyDown(KeyCode.F2))
                {
                    return false;
                }
                else
                {
                    return true;
                }
            }

            return false;
        }

        public int GetFrameAdvantage(bool getP1)
        {

            var p1FrameLeft = fighter1.currentActionFrameCount - fighter1.currentActionFrame;
            if (fighter1.isAlwaysCancelable)
                p1FrameLeft = 0;

            var p2FrameLeft = fighter2.currentActionFrameCount - fighter2.currentActionFrame;
            if (fighter2.isAlwaysCancelable)
                p2FrameLeft = 0;

            if (getP1)
                return p2FrameLeft - p1FrameLeft;
            else
                return p1FrameLeft - p2FrameLeft;
        }
    
        /* 
        writes the training data to the correct file
        marks data as having been logged and labels file with date & time
        called at each game tick
        */ 
        private void OutputTrainingData()
        {
            // unreachable code past the return, remove this return statement
            //  to reenable data logging
            return;

            string trainingDataLogPath = "dataLog#" + 
            DateTime.Now.ToString(@"MM-dd-yy--HH-mm-ss") + 
            ".txt";

            try
            {
                
            var path = @"TrainingData/" + trainingDataLogPath;
            File.WriteAllText(path, trainingData);

            trainingData = "";

            dataLogged = true;
            }
            catch
            {
                UnityEngine.Debug.Log("Unable to write to log file");
            }
        }

        /*
        updates the message queue to be sent to the server with a message of
            type string
        simultaneously releases the "messageAvailable" semaphore
        */
        public void UpdateMessageQueue(string message)
        {
            if (frameCount % 4 == 0){
                // currently sending a dummy message until further notice
                messageQueue.Enqueue(message);
                messageAvailable.Release();}
                }

    }
}