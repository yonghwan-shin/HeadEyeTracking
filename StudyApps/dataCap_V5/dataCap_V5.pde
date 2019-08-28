// you need to set
int subNum      = 16;               // the participant number
String remoteIP = "192.168.0.4";  // the HMD system's IP address (HoloLens) on the local router
String statusIP = "192.168.0.3"; // the monitoring PCs IP address  on the local router
String eyeCam   = "pupil.1";       //  equals pupil.0 or pupil.1
int trialCount=0;

//1. Ask to borrow 200Hz camera.

//IMU. Data is left/right, up/down. Left is smaller, up is smaller
//Pupil. Data is left/right, up/down. Left is bigger, up is bigger

//1. get pupil data from xml (more accurate - just 2D for now)
//2. normalize imu data and flip/normalize pupil data
//3. add them and see how it looks - if flat...good. 

import processing.sound.*;

// where the pupil contents is read
pupilThread p; 
// where we store all the data
volatile ArrayList<String> pupilData = new ArrayList<String>();
volatile ArrayList<String> imuData = new ArrayList<String>();
// central timing variables we use to tag packets. 
volatile long pupilTimeStamp = 0; // updated when we get pupil data
volatile long imuTimeStamp   = 0; // updated when we get imu data

volatile boolean logData    = false;  // whether or not we are in a trial.
volatile long logDataTime   = -1;
volatile long logDataDur    = -1;

//boolean usePupil = false;
//boolean useIMU   = false;

boolean usePupil = true;
boolean useIMU   = true;

boolean useAudio = false;
boolean useUDP   = true;


long  UDPSynchTime       = -1;// the time we heard the synch beep from the other device (e.g. hololens); 
boolean handshake = false;
boolean send = true;
long subTimeStamp=0;
String localIP             = "N/A"; // we detect this (if needed)


String getFN() {
  return myStudy.getFN();
}

String getSummary() {
  return myStudy.getFN();
}

// noises for feedback
TriOsc triOsc;
Env env;
float attackTime = 0.001;
float sustainTime = 0.004;
float sustainLevel = 0.3;
float releaseTime = 0.4;


final int MODE_START     = 0;  // waiting for manual start from the study instructor (a key)
final int MODE_SENDTRIAL = 1;  // we send trial details to the other device
final int MODE_WAITBEEP  = 2;  // we wait for the beep indicating we ready to go - we need a timeout here. 
final int MODE_LOGGING   = 3;  // after the beep, we just log data
final int MODE_PAUSE     = 4;  // after the beep, we just log data
final int MODE_END       = 5;  // all over

long beepTimeout = 2000;// if nothing happened after 10 seconds, we try again.   
long recordWait  = 6500; // we record 
long endWait     = 2000; // we wait at the end before quiting
long pauseTimeout= 1000; // how long we wait here on the PC to make sure we are synched with the other device

long modeChange  = 0; 
int mode = MODE_START; 



void setup()
{
  //size(600, 200);
  size(600, 600, P3D);
  textSize(48); 

  myStudy = new study(subNum); 
  String s = getTrial();

  /*
  // dump the trials to test them
   while (s!=null)
   {
   println(s, myStudy.lastTrial);
   s = getTrial();
   }
   */


  localIP = getIPAddress(); 
  println("Local PC IP address is " + localIP); 

  // setup up noises for feedback
  triOsc = new TriOsc(this);   // Create triangle wave
  env  = new Env(this);        // Create the envelope 

  if (useUDP)
    initUDP(); 

  if (usePupil)
  {
    p = new pupilThread(this); 
    p.start();
  }

  if (useIMU)
    initIMU(); 

  if (useAudio)
    initAudio(); 

  frameRate(30);
  textAlign(CENTER, CENTER);
  //fill(0); 
  println("Done setup");
}


void draw()
{
  background(255);  
  long now = millis(); 

  // process thread data here. 
  if (usePupil)
  {
    String local = ""; 
    double localConf = -1;
    int localRate = -1;
    synchronized (this) 
    {
      local = lastPupil;

      // manage status messages
      if (!sentPupilConf)
      {
        localConf = globalPupilConf;
        sentPupilConf = true;
      }
      if (!sentPupilRate)
      {
        localRate = globalPupilRate;
        sentPupilRate = true;
      }
    }; 

    // send the safe non-synch'd/shared copy 
    if (localConf!=-1)
      sendStatus("Pupil confidence: " + localConf);
    if (localRate!=-1)
      sendStatus("Pupil Rate: " + localRate);
  }


  if (useIMU)
  {
    String local = ""; 
    int localRate = -1;
    synchronized(this) 
    {
      local = lastIMU;

      // manage status messages
      if (sendIMURate)
      {
        localRate = globalIMURate;
        sendIMURate = false;
      }
    }; 

    // send the safe non-synch'd/shared copy 
    if (localRate!=-1)
      sendStatus("IMU rate: " + localRate);

    pushMatrix();
    Matrix4x4 m = drawCube(local);
    popMatrix();

    PVector pIntersect = null; 

    pushMatrix();
    if (m!=null)
    {
      Vec3D vec = m.applyTo(new Vec3D(0, 200, 0));
      //translate(width/4, height/4*3, 0);
      translate(width/2, height/2, 0);
      pushMatrix();
      fill(128, 32);
      translate(0, 0, -96.5);  
      box(500, 500, 1);
      fill(0); 
      popMatrix(); 
      rotateX(-PI/2);
      strokeWeight(10);
      stroke(255, 0, 0); 
      line(0, 0, 0, vec.x, vec.y, vec.z);
      strokeWeight(1);
      stroke(0); 

      pIntersect = Intersect(new PVector(vec.x, vec.y, vec.z), new PVector(0, 0, 0), new PVector(0, -1, 0), new PVector(0, 100, 0));

      translate(pIntersect.x, pIntersect.y, pIntersect.z);

      sphere(5);
    }
    popMatrix(); 

    /*
    if (pIntersect!=null)
     {
     text(round(pIntersect.x*1000.0)/1000.0, width/4, height/10*9);
     text(round(pIntersect.z*1000.0)/1000.0, width/4*3, height/10*9);
     }
     */
    fill(255);
  }


  textSize(16);   
  if (mode == MODE_START)
  {
    fill(0);
    text("Press a key to start\n" + getSummary(), width/2, height/12); 
    if (subTimeStamp+3000<now) {
      sendMsg("sub"+subNum);
      println("sending sub");
      subTimeStamp= now;
    }
    if (UDP_START !="") {
      if (UDP_START.substring(0, 3).equals("sub")) {
        handshake = true;
        println("handshake completed. delay is : "+ (now- subTimeStamp)+"... click space bar to start");
        UDP_START = "";
      }
    }
  } else if (mode == MODE_SENDTRIAL) // no waiting here - its just once.
  {
    myStudy.addTiming(now); 
    String s = myStudy.getFN(); 
    sendMsg(s);
    println("Sending trial details: " + s); 
    if (useAudio)
      listenAudio();  
    mode = MODE_WAITBEEP; 
    modeChange = now;
  } else if (mode == MODE_WAITBEEP)
  {
    if (UDP_START.length()>3 &&UDP_START.substring(0, 5).equals("START"))
    {
      //UDP_START="";
      UDPSynchTime = System.currentTimeMillis();


      println("Synch beep detected at " + UDPSynchTime); 

      if (isBreakTrial(myStudy.lastTrial)) // this is a break trial, so process the beep to move direct to the next trial
      //if (UDP_START.equals("START_BREAK"))
      {
        if (getTrial() == null)
        {
          mode = MODE_END; 
          modeChange = now; 
          return;
        }
        mode = MODE_PAUSE; 
        modeChange = now;
      } else if (UDP_START.equals("START_CENTER")) // we move on to logging
      {
        synchronized(this) {
          logData = true;
        } 
        logDataTime = now; 
        mode = MODE_LOGGING;
      } else {
        println("ERROR");
      }
      modeChange = now;
    } else if (now - modeChange > beepTimeout) // try sending again. 
    {
      mode = MODE_SENDTRIAL; 
      modeChange = now;
    }
    UDP_START="";
    fill(0);
    text("Waiting for synch", width/2, height-height/20);
  } else if (mode == MODE_LOGGING)
  {

    text("Recording\n" + getSummary(), width/2, height/12);
    if (now - modeChange > recordWait)
    {
      synchronized(this) {
        logData = false;
      }
      logDataDur = now - logDataTime;
      saveData(); 
      logDataTime = -1;
      logDataDur  = -1; 
      synchronized(this) {
        pupilData.clear();
      }
      synchronized(this) {
        imuData.clear();
      }

      // increment study count and check for end of study
      if (getTrial() == null)
      {
        mode = MODE_END; 
        modeChange = now; 
        return;
      }
      mode = MODE_PAUSE; 
      modeChange = now;
      println("Current trial is ... " + (++trialCount));
    }
    fill(0);
    text("Logging", width/2, height-height/20);
  } else if (mode == MODE_PAUSE)
  {
    if (now - modeChange > pauseTimeout) // try sending again. 
    {
      mode = MODE_SENDTRIAL; 
      modeChange = now;
    } else
      text("Pause", width/2, height/12);
  } else if (mode == MODE_END)
  {
    fill(0);
    text("Study Finished", width/2, height-height/20);
    if (now - modeChange > endWait)
    {
      sendMsg("END");  
      if (usePupil)
        p.quit();// quit the pupil thread
      println("Study Finished");
      exit();
    }
  }


  text("Local: " + localIP + ", Remote: " + remoteIP, width/2, height-height/10);
}


void keyPressed()
{
  long now = millis();

  if (key == ESC)
  {
    println("Quitting!");
    if (usePupil)
      p.quit();// quit the pupil thread
    exit();
  } else if (key== ' ')
  {
    if (mode == MODE_START)
    {
      mode = MODE_SENDTRIAL; 
      modeChange = now;
    }
  }
}


void saveData()
{
  String fn = "Processing_"+subNum + "/" + getFN(); 
  println("Saving " + getSummary() + ": " + pupilData.size() + " eye and " + imuData.size() + " head packets to " + fn); 

  String[] data = new String[pupilData.size() + imuData.size() + 2]; 
  data[0] = fn;
  data[1] = "UDPSynch," + UDPSynchTime + ",pupilPackets," + pupilData.size()+ ",imuPackets," + imuData.size() +
    ",logDataStart," + logDataTime +
    ",logDataEnd," + millis() +
    ",logDataDur," + logDataDur; 
  int count = 2; 
  for (String s : pupilData)
  {
    data[count] = s; 
    count++;
  }
  for (String s : imuData)
  {
    data[count] = s; 
    count++;
  }
  saveStrings(fn, data);
}



// the beeps
void deepBeep()
{
  triOsc.freq(200);
  triOsc.play();
  env.play(triOsc, attackTime, sustainTime, sustainLevel, releaseTime);
}

void highBeep()
{
  triOsc.freq(500);
  triOsc.play();
  env.play(triOsc, attackTime, sustainTime, sustainLevel, releaseTime);
}

void shortBeep()
{
  triOsc.freq(500);
  triOsc.play();
  env.play(triOsc, attackTime, sustainTime/4, sustainLevel/4, releaseTime/4);
}
