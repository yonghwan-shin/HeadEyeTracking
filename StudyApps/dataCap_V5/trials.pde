import java.util.ArrayList; 

// stand/walk
final int NUM_TARGETS     = 8; // 8 targets - for test, use 1
final int NUM_TYPES       = 2; // two target types: world/UI
final int NUM_POSES       = 2; // two poses: stand/walk

// defs for types/poses 
final int TYPE_WORLD      = 0;
final int TYPE_UI         = 1;

final int POSE_STAND      = 0;
final int POSE_WALK       = 1;

// 2x2 design
// stand -> walk
//   World->UI   = 4
//   UI->World   = 4
// walk -> stand
//   World->UI   = 4
//   UI->World   = 4

// 4 conditions x 8 targets x 5 blocks x 10 seconds = 1600 seconds or 26 mins 40 seconds. 

int NUM_REPS               = 5; // 5 reps - for test, use 1 

// GOALS: 
// 1 prove its very hard. 
// 2 get data to inform correction designs. 

class trial
  {
  int target; 
  int env; 
  int pose;
  
  trial(int tar, int en, int pos)
    {target = tar; env = en; pose = pos;}
   
  
  String getDisc() {return "T" + target + "_" + (env==TYPE_WORLD ? "EW" : "EU") +"_"+ (pose==POSE_STAND ? "PS" : "PW");}
  }
  
class block
  {
  ArrayList<trial> trials;
  int bID; 
  
  // add a regular block
  block(int e, int p, int BID) 
    {
    bID = BID;
    trials = new ArrayList<trial>();
    for (int i=0;i<NUM_TARGETS;i++)
      trials.add(new trial(i, e, p));  
    Collections.shuffle(trials); 
    }
    
  // we use this for special trials (e.g. breaks)  
  block(int t, int e, int p, int BID) 
    {
    bID=BID; 
    trials = new ArrayList<trial>();
    for (int i=0;i<1;i++)
      trials.add(new trial(t, e, p));   
    }
  }
  
class study
  {
  ArrayList<block> blocks;
  
  String lastTrial; 
  long timing; 
  
  String startupTimeStamp;
  
  int pNum; 
  
  void updateDisc(String s)
    {
    lastTrial = s;
    lastTrial += "_B" + blocks.get(0).bID; 
    lastTrial += "_C" + (NUM_TARGETS-blocks.get(0).trials.size()); 
    }
    
  void addTiming(long now)
    {
    timing = now;
    }
  
  String getFN()
    {
    return lastTrial + "_P" + timing + "_S" + pNum + "_" + startupTimeStamp + ".csv"; 
    } 
  
  study(int PNum)
    {
    pNum = PNum; 
    startupTimeStamp = getDateTS(); 
    
    lastTrial = null;   
    blocks = new ArrayList<block>();
    if (PNum%4 == 0)
      {
      blocks.add(new block(-1, TYPE_WORLD, POSE_STAND, -1)); // this is a pause
      for (int i=0;i<NUM_REPS;i++)  blocks.add(new block(TYPE_WORLD,  POSE_STAND, i));
      for (int i=0;i<NUM_REPS;i++)  blocks.add(new block(TYPE_UI,     POSE_STAND, i));
      blocks.add(new block(-1, TYPE_WORLD, POSE_WALK, -1)); // this is a pause
      for (int i=0;i<NUM_REPS;i++)  blocks.add(new block(TYPE_WORLD,  POSE_WALK, i));
      for (int i=0;i<NUM_REPS;i++)  blocks.add(new block(TYPE_UI,     POSE_WALK, i));
      }
    else if (PNum%4 == 1)
      {
      blocks.add(new block(-1, TYPE_WORLD, POSE_WALK, -1)); // this is a pause
      for (int i=0;i<NUM_REPS;i++)  blocks.add(new block(TYPE_WORLD,  POSE_WALK, i));
      for (int i=0;i<NUM_REPS;i++)  blocks.add(new block(TYPE_UI,     POSE_WALK, i));
      blocks.add(new block(-1, TYPE_WORLD, POSE_STAND, -1)); // this is a pause
      for (int i=0;i<NUM_REPS;i++)  blocks.add(new block(TYPE_WORLD,  POSE_STAND, i));
      for (int i=0;i<NUM_REPS;i++)  blocks.add(new block(TYPE_UI,     POSE_STAND, i));
      }
    else if (PNum%4 == 2)
      {
      blocks.add(new block(-1, TYPE_UI, POSE_STAND, -1)); // this is a pause
      for (int i=0;i<NUM_REPS;i++)  blocks.add(new block(TYPE_UI,     POSE_STAND, i));
      for (int i=0;i<NUM_REPS;i++)  blocks.add(new block(TYPE_WORLD,  POSE_STAND, i));
      blocks.add(new block(-1, TYPE_UI, POSE_WALK, -1)); // this is a pause
      for (int i=0;i<NUM_REPS;i++)  blocks.add(new block(TYPE_UI,     POSE_WALK, i));
      for (int i=0;i<NUM_REPS;i++)  blocks.add(new block(TYPE_WORLD,  POSE_WALK, i));
      }
    else if (PNum%4 == 3)
      {
      blocks.add(new block(-1, TYPE_UI, POSE_WALK, -1)); // this is a pause
      for (int i=0;i<NUM_REPS;i++)  blocks.add(new block(TYPE_UI,     POSE_WALK, i));
      for (int i=0;i<NUM_REPS;i++)  blocks.add(new block(TYPE_WORLD,  POSE_WALK, i));
      blocks.add(new block(-1, TYPE_UI, POSE_STAND, -1)); // this is a pause
      for (int i=0;i<NUM_REPS;i++)  blocks.add(new block(TYPE_UI,     POSE_STAND, i));
      for (int i=0;i<NUM_REPS;i++)  blocks.add(new block(TYPE_WORLD,  POSE_STAND, i));
      }
    }
  }


study myStudy;  

String getTrial()
  {
  if (myStudy.blocks.size()>0)
    {
    if (myStudy.blocks.get(0).trials.size()>0)
      {
      String s = myStudy.blocks.get(0).trials.get(0).getDisc();
      myStudy.updateDisc(s); 
      
      // remove the trial
      myStudy.blocks.get(0).trials.remove(0);
      // if last trial in block, remove block
      if (myStudy.blocks.get(0).trials.size()==0)
        myStudy.blocks.remove(0); 
      
      return s; 
      }
    }
  return null; 
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
  
boolean isBreakTrial(String s)
  {
  String[] parts = s.split("_"); 
  if (Integer.parseInt(parts[0].substring(1)) == -1)
    return true;
  return false;
  }
