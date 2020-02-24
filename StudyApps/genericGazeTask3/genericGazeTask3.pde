import java.util.Collections;

// update all these:
int subNum                 = 208;                    // update as needed        
boolean usePupil           = true;                // turn both to true to turn on sensing!
boolean useIMU             = true;
float thirtyDiagonalSquareWidthInPixels  =   821;//821;  // whatever the final value is
String eyeCam              = "pupil.1";            //  equals pupil.0 or pupil.1

// possible modify these - the sizes of jump between trials (in degrees of visual angle)
int shortJump = 6; 
int medJump   = 10;
int longJump  = 14;

// possible modify these - these are the speeds of moving targets (in degrees of visual angle/second)
int slowSpeed = 10;
int medSpeed  = 20;
int fastSpeed = 30;
// no updating below here. 


float maxRect         = 10.6066;          // the x/y dims of a sqaure with 30 degrees of visual angle along its diagonal 
float pixelsPerDegree = thirtyDiagonalSquareWidthInPixels/(maxRect*2);  // the number of pixels in a single degree of visual angle. 

int w      = (int)(pixelsPerDegree*25.0);         // these should resemble HoloLens FoV
int h      = (int)(pixelsPerDegree*25.0);         // FYI: we use 25 to provided a reasonable edge gap around the 21.2132 square FoV needed for 30 FoV diagonal. 


// where the pupil contents is read
pupilThread p; 
// where we store all the data
volatile ArrayList<String> pupilData = new ArrayList<String>();
volatile ArrayList<String> imuData = new ArrayList<String>();
ArrayList<String> screenData = new ArrayList<String>();
// central timing variables we use to tag packets. 
volatile long pupilTimeStamp = 0; // updated when we get pupil data
volatile long imuTimeStamp   = 0; // updated when we get imu data

long  audioSynchTime       = -1;// the time we heard the synch beep from the other device (e.g. hololens);

volatile boolean logData    = false;  // whether or not we are in a trial.

float saccadesFix= 750;  // most fixations between saccades are 750ms
float pursuitsFix= 1000; // fixations before/after putsuits are 1000ms
float expDuration= 5000; // free exploration at the study start
float setDuration= 2000; // focus on the center...



int targetSize       = (int)pixelsPerDegree;    // how big the target is - 1 degree

long endTime     = 2500; // how long we wait at the end

boolean started = false; // whether or not we are beginning

int framerate    = 60;

long start       = 0;    // helper var for starttime
float posX;              // current X
float posY;              // current Y
float lastPosX;          // last X
float lastPosY;          // last Y
float rotCenterX;        // rot center ((just center of lastpoint to point really)
float rotCenterY;        //
float rotDist;           // rot distance (circular rads)
float rotAngle;          // rot angle (just angle from lastpoint to point really)
int rotInvert;           // clockwise or counter

// intermediary timing vars
long duration;           // step/trial duration (for fixations)
float purTime        = 0;// we calc how long a pursuit will take to lerp its drawing wrt time passed. Smoother than simple increment.  

// order of trials/steps
int EXP = 0;
int SET = 1;
int FIX = 2;
int PUR = 3;
int CIR = 4;

boolean logConfTotal = true; 
double confTotal = 0; 
double confSamples = 0; 

class trial
{
  int mode;
  float param1; // amp/radius
  float param2; // velocity

  trial(int m, float p1, float p2)
  {
    mode = m;
    param1 = p1;
    param2 = p2;
  }
}

ArrayList<trial> trials;   
int modeIndex = 0;       // current step/trial 

void settings()
{
  size(w, h);
  smooth(8);
}

// init everything
void setup()
{
  targetSize = (int)pixelsPerDegree; 
  


  trials = new ArrayList<trial>();
  // saccades - use three reps of the three used
  trials.add(new trial(FIX, shortJump, -1)); 
  trials.add(new trial(FIX, shortJump, -1)); 
  trials.add(new trial(FIX, shortJump, -1)); 

  trials.add(new trial(FIX, medJump, -1)); 
  trials.add(new trial(FIX, medJump, -1)); 
  trials.add(new trial(FIX, medJump, -1));
  
  trials.add(new trial(FIX, longJump, -1)); 
  trials.add(new trial(FIX, longJump, -1)); 
  trials.add(new trial(FIX, longJump, -1));



  // staight pursuits - use the 9 smallest (remove three really large ones)
  trials.add(new trial(PUR, shortJump, slowSpeed)); 
  trials.add(new trial(PUR, shortJump, medSpeed)); 
  trials.add(new trial(PUR, shortJump, fastSpeed)); 

  trials.add(new trial(PUR, medJump, slowSpeed)); 
  trials.add(new trial(PUR, medJump, medSpeed)); 
  trials.add(new trial(PUR, medJump, fastSpeed));

  trials.add(new trial(PUR, longJump, slowSpeed)); 
  trials.add(new trial(PUR, longJump, medSpeed)); 
  trials.add(new trial(PUR, longJump, fastSpeed));

  // circular pursuits - use all 9 - all full 360 spins in the original paper
  trials.add(new trial(CIR, shortJump, slowSpeed)); 
  trials.add(new trial(CIR, shortJump, medSpeed)); 
  trials.add(new trial(CIR, shortJump, fastSpeed)); 

  trials.add(new trial(CIR, medJump, slowSpeed)); 
  trials.add(new trial(CIR, medJump, medSpeed)); 
  trials.add(new trial(CIR, medJump, fastSpeed));

  trials.add(new trial(CIR, longJump, slowSpeed)); 
  trials.add(new trial(CIR, longJump, medSpeed)); 
  trials.add(new trial(CIR, longJump, fastSpeed));

  // jumble it up 
  Collections.shuffle(trials); 

  // add in the start
  trials.add(0, new trial(SET, -1, -1));
  trials.add(0, new trial(EXP, -1, -1));

  if (usePupil)
  {
    p = new pupilThread(this); 
    p.start();
  }

  if (useIMU)
    initIMU(); 

  frameRate(framerate); 
  textAlign(CENTER, CENTER);
}



void draw()
{
  long now = millis(); 
  background(0);
  stroke(128);
  noFill();
  //rect(width/2-maxRect*pixelsPerDegree, height/2-maxRect*pixelsPerDegree, maxRect*pixelsPerDegree*2, maxRect*pixelsPerDegree*2);   
  fill(255); 

  if (!started)
  {
    String s = "In this task, you will first see four white circles for five seconds.\n\n";
    s+= "Please look at any of these circles freely.\n\n";
    s+= "You will then see a single dot for the rest of the session (around 1 minute).\n\n";
    s+= "It will move to different locations instantly, by moving in a straight line\n\n";
    s+= "and by moving in a circular pattern.\n\n";
    s+= "Your task is to LOOK AT THE SINGLE DOT\n\nby following it closely as possible with YOUR EYES AT ALL TIMES.\n";
    text(s, width/2, height/4);
    
    text("Press a key to start.", width/2, height/4*3);
    return;
  }

  // quit at end?
  if (modeIndex>=trials.size())
  {
    text("Session complete.", width/2, height/2); 
    if (now-start > endTime)
      {
      println("Mean confidence in the whole study was " + confTotal/confSamples); 
      exit();
      }
    return;
  }


  if ( now>start + duration)  // this is EXP, SET, FIX
  {
    // store last positions for pursuit
    lastPosX = posX;
    lastPosY = posY;

    if (modeIndex+1<trials.size())
    {
      if (trials.get(modeIndex+1).mode==SET) // set to middle on first trial
      {
        posX = width/2;
        posY = height/2;
        duration = (long)setDuration;
      } 
      
      
      else if (trials.get(modeIndex+1).mode==FIX ) // we have distances of 6, 11, 14 degrees in random dirs
      {        
        PVector iP = generateInternalPoint(trials.get(modeIndex+1), lastPosX, lastPosY, false);
        posX = iP.x; posY = iP.y;
        duration = (long)saccadesFix;
      } 
      
      
      else if (trials.get(modeIndex+1).mode==PUR ) // we have distances of 6, 12, 22, 28 degrees in random dirs and speeds of 10, 20, 30
      {
        PVector iP = generateInternalPoint(trials.get(modeIndex+1), lastPosX, lastPosY, false);
        posX = iP.x; posY = iP.y;
        float speed = trials.get(modeIndex+1).param2; // this is in degrees/second
        speed       = speed * pixelsPerDegree;        // now in pixels per second
        purTime     = (dist(lastPosX, lastPosY, posX, posY)/speed)*1000.0; // this is in ms
        duration    = (long)purTime + (long)pursuitsFix;
      } 
      
      
      else if (trials.get(modeIndex+1).mode==CIR ) // we have radii of 6, 8, 14 degrees in random dirs at speeds of 18, 25, 44 (angular velocity of 180)
      {
        rotCenterX = rotCenterY = rotDist = rotAngle = 0;
        PVector iP = generateInternalPoint(trials.get(modeIndex+1), lastPosX, lastPosY, true);
        posX = iP.x; posY = iP.y;
        
        /*float vX, vY;
        vX = vY = 
        // check we can see everything. 
        
        int count = 0; 
        while (count==0 || rotCenterX-rotDist<targetSize/2 || rotCenterX+rotDist>width-targetSize/2 || rotCenterY-rotDist<targetSize/2 || rotCenterY+rotDist>height-targetSize/2)
          {
          PVector iP = generateInternalPoint(trials.get(modeIndex+1), lastPosX, lastPosY, true);
          posX = iP.x; posY = iP.y;
        
          // look at the vector and figure out where we can rotate a circle 
          vX = (posX-lastPosX)/2;
          vY = (posY-lastPosY)/2;
          rotCenterX = lastPosX + vX;
          rotCenterY = lastPosY + vY;
          rotDist = dist(0,0,vX,vY);    
          rotAngle= atan2(-vY,-vX); 
          count++;
          if (count>10)
            println("WARNING: struggling to guess a good location for a circular pursuit. Continuning...", count, millis());
          else if (count>50)
            {
            println("ERROR: cannot guess a good location for a circular pursuit after", count, "attempts. Quitting.");
            exit();
            }
          }
          */
        
        //println("Made", count, "guesses for the circle"); 
        
        // calculate the speed of rotation 
        float speed = trials.get(modeIndex+1).param2; // this is in degrees of visual angle/second
        speed       = speed * pixelsPerDegree;        // now in pixels per second
        speed       = (2 * PI * rotDist)/speed;       // the number of pixels in the circumference, divided by the speed - the number of seconds it will take... 
        purTime     = speed * 1000.0;                 // the amount of time for 360 degrees... 
        duration    = (long)purTime + (long)pursuitsFix;
        rotInvert   = (int)random(0, 2); 
        
        // swap the pos's
        float t = lastPosX; lastPosX = posX; posX = t;
              t = lastPosY; lastPosY = posY; posY = t;
        
      }
    }

    // move through the trials
    modeIndex = modeIndex+1; 
    start = now;
  }


  // this will only happen once at the end...
  if (modeIndex>=trials.size())
  {
    synchronized(this) {
      logData = false;
    }
    saveData(); 
    synchronized(this) {
      pupilData.clear();
    }
    synchronized(this) {
      imuData.clear();
    }
    return;
  }


  String details = trials.get(modeIndex).mode + "," + trials.get(modeIndex).param1 + "," + trials.get(modeIndex).param2;  
  if (trials.get(modeIndex).mode==EXP) 
  {
    float dev = maxRect * pixelsPerDegree;
    float cx  = width/2;
    float cy  = height/2;
    ellipse(cx-dev, cy-dev, targetSize, targetSize);
    ellipse(cx-dev, cy+dev, targetSize, targetSize);
    ellipse(cx+dev, cy-dev, targetSize, targetSize);
    ellipse(cx+dev, cy+dev, targetSize, targetSize);
    screenData.add(now +","+ (int)(cx-dev) +","+ (int)(cy-dev) +","+ targetSize +","+ width +","+ height +",EXP," + details);
  } else if (trials.get(modeIndex).mode==SET) 
  {
    posX = width/2; 
    posY = height/2; 
    ellipse(posX, posY, targetSize, targetSize);
    screenData.add(now +","+ (int)posX +","+ (int)posY +","+ targetSize +","+ width +","+ height +",SET," + details);
  }  

  //render a fixation  
  else if (trials.get(modeIndex).mode==FIX) 
  {
    ellipse(posX, posY, targetSize, targetSize);
    screenData.add(now +","+ (int)posX +","+ (int)posY +","+ targetSize +","+ width +","+ height+",FIX," + details);
  }

  // render a pursuit
  else if (trials.get(modeIndex).mode==PUR)
  {
    float localPosX = lastPosX;
    float localPosY = lastPosY;
    if (lastPosX != posX && lastPosY != posY)
    {
      if (now-start < purTime)
      {
        localPosX = lerp(lastPosX, posX, (float)(now-start) / purTime);
        localPosY = lerp(lastPosY, posY, (float)(now-start) / purTime);
      } else 
      {
        lastPosX = localPosX = posX;
        lastPosY = localPosY = posY;
        start = now;
      }
    }
    ellipse(localPosX, localPosY, targetSize, targetSize); 
    screenData.add(now +","+ (int)localPosX +","+ (int)localPosY +","+ targetSize +","+ width +","+ height+",PUR," + details);
  }
  
  // render a circular pursuit
  else if (trials.get(modeIndex).mode==CIR)
  {    
    float localPosX = posX;
    float localPosY = posY;
    if (now-start < purTime)
      {
      float angle = lerp(0, 360, (float)(now-start) / purTime);
      if (rotInvert==1)
        angle = lerp(360, 0, (float)(now-start) / purTime);
      localPosX = rotCenterX + (rotDist * cos(radians(angle) + rotAngle));
      localPosY = rotCenterY + (rotDist * sin(radians(angle) + rotAngle));
      } 
    
    ellipse(localPosX, localPosY, targetSize, targetSize);  
    //println(rotAngle, angle, rotDist, rotCenterX, rotCenterY);
    screenData.add(now +","+ (int)localPosX +","+ (int)localPosY +","+ targetSize +","+ width +","+ height+",CIR," + details);
  }
}


void keyReleased()
{
  if (key==' ')
  {
    duration = (long)expDuration;
    start = millis(); 
    started = true;
    synchronized(this) {
      logData = true;
    }
  }
}


String getDateTS()
{
  String spacer = "";
  String ret = java.time.LocalDate.now().toString(); 
  ret = ret.substring(ret.indexOf("-")+1);
  ret = ret.replace("-", spacer);

  String ret2 = java.time.LocalTime.now().toString();
  ret2 = ret2.replace(":", spacer); 
  ret2 = ret2.substring(0, ret2.indexOf(".")); 

  return ret+spacer+ret2;
}

void saveData()
{
  String fn = "Sub" + subNum + "_" + getDateTS() + "_configTestSensors.csv";  

  println("Saving " + pupilData.size() + " pupil packets");
  println("Saving " + imuData.size() + " IMU packets");
  println("Saving " + screenData.size() + " screen packets");

  String[] data = new String[pupilData.size() + imuData.size() + screenData.size() + 2]; 
  data[0] = fn;
  data[1] = "AudioSynch," + audioSynchTime; 
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
  for (String s : screenData)
  {
    data[count] = s; 
    count++;
  }
  saveStrings(fn, data);
}

float mod = 5; 
float guesses = 25;
PVector generateInternalPoint(trial t, float lPX, float lPY, boolean checkCircle)
{
  float pixelDist = pixelsPerDegree*t.param1; 
  float dir = random(0, 360); 
  float pX = lPX + pixelDist * cos(radians(dir));
  float pY = lPY + pixelDist * sin(radians(dir));
  
  float d = maxRect*pixelsPerDegree;
  
  int count = 1; 
  boolean passed = false;
  boolean failed = false;
  while (!passed && !failed)
    {
    if (count>=guesses) // first make <guesses> random guesses 
      {
      // if those fail, increment from a line to center in steps of <mod>
      dir = degrees(atan2(height/2-lPY, width/2-lPX)) + (count-guesses)*mod;
      println("Warning: struggling to guess a good location for target - tried", count, "times. Aiming to center at", degrees(atan2(height/2-lPY, width/2-lPX)), "plus mod of", ((count-guesses)*mod));
      if (count>guesses + (360/mod)-1)
        {
        println("ERROR: Permanent failure after", count, "random and sequential guesses. Quitting.");
        failed = true; 
        }
      }
    else
      dir = random(0, 360);
    pX = lPX + pixelDist * cos(radians(dir));
    pY = lPY + pixelDist * sin(radians(dir)); 
    count++; 
    
    passed = abs(pX-width/2)<d && abs(pY-height/2)<d; // check for point within boundaries
    if (checkCircle) // additional check for circle within boundaries
      {
      // look at the vector and figure out where we can rotate a circle 
      float vX = (pX-lPX)/2;
      float vY = (pY-lPY)/2;
      rotCenterX = lastPosX + vX;
      rotCenterY = lastPosY + vY;
      rotDist = dist(0,0,vX,vY);    
      rotAngle= atan2(-vY,-vX); 
      if (rotCenterX-rotDist<targetSize/2 || rotCenterX+rotDist>width-targetSize/2 || rotCenterY-rotDist<targetSize/2 || rotCenterY+rotDist>height-targetSize/2)
        passed = false;
      }
    }
  
 //println("Made", count, "guesses for the end point"); 
 if (failed) 
   exit(); 
 //else we have passed
 return new PVector (pX, pY); 
}
