import org.zeromq.ZMQ;    // not needed - handled automatically because jar is in the code folder...
import java.nio.charset.StandardCharsets; // can we remove? Get the constant from ZMQ?


volatile String lastPupil = "";  


volatile boolean sendPupilRate = false;
volatile boolean sentPupilRate = false;
volatile int globalPupilRate = 0;


volatile boolean sendPupilConf = false;
volatile boolean sentPupilConf = false;
volatile double globalPupilConf = 0;




class pupilThread extends Thread {  
  private volatile boolean quit = false;

  ZMQ.Context ctx;
  ZMQ.Socket s;  

  MessagePack msgpack = new MessagePack(); // for the serialized msg format: https://github.com/msgpack/msgpack-java
  boolean pupilReady = false;

  int packetCount = 0; 
  long currentSecond = 0;  
  long last200ms = 0;

  PApplet parent; 

  pupilThread(PApplet parIn)
  {
    parent = parIn;
  }

  public void start()
  {
    initPupil(); 
    super.start();
  }

  public void quit()
  {
    quit = true;
  }

  public void run()
  {
    println("Pupil reading thread starting up.");

    while (!pupilReady); // make sure everything is init'd

    while (!quit) {
      getData();         // just grab data
    }

    println("Pupil reading thread shutting down.");
  }


  void initPupil()
  {
    // open up a new communication channel to the remote to request the current port number.   
    ctx = ZMQ.context (1);
    s = ctx.socket (ZMQ.REQ);
    s.connect("tcp://127.0.0.1:50020");
    s.send("SUB_PORT");
    byte[] reply = s.recv(0);
    String port = new String(reply, StandardCharsets.UTF_8); 
    println("Pupil is broadcasting on " + port);
    s.disconnect("tcp://127.0.0.1:50020"); 

    // set up subscriber
    s = ctx.socket (ZMQ.SUB);
    // ideally implement a check/monitor on the connection - TODO
    // but, regardless, connect to the local broadcast. 
    s.connect("tcp://127.0.0.1:" + port);

    // ask for the data we want - "pupil.0" or "pupil.1" for raw data from the eye cams - good for smooth moves. 
    s.subscribe(eyeCam.getBytes());

    pupilReady = true;
  }

  String getData()
  {
    byte[] b = new byte[2048];
    b = s.recv();   
    processing.data.JSONObject pupilJson = null; // need to use full class path becuase somewhere in the mess of libs for zmq, we have a similarly named object
    String raw = "empty" ;
    
    try {
      raw = String.format("%s", msgpack.read(b));
      pupilJson = parseJSONObject(raw);
    } 
    catch (Exception e) {
    }
    if (pupilJson!=null)
    {  
      long now = millis(); 
     
      synchronized (parent) {pupilTimeStamp = now;}
      
      int pupilRate = -1;
      if (now/1000>currentSecond)
      {
        sendPupilRate = true; 
        sentPupilRate = false; 
        pupilRate = packetCount+1;
        //println("Pupil Rate: " + pupilRate + " Hz");
        //sendStatus("Pupil Rate: " + pupilRate + " Hz");
        
        currentSecond = now/1000;
        packetCount = 0; 
      } else
        packetCount++; 

      long current200ms  = (now%1000)/200; // five times a second...
      double pupilConf = 0;
      if (current200ms!=last200ms)
        {
        sendPupilConf = true; 
        sentPupilConf = false; 
        last200ms     = current200ms;
        pupilConf     = pupilJson.getDouble("confidence"); 
        }
      
      if (logConfTotal)
        {
        confTotal += pupilJson.getDouble("confidence");
        confSamples ++; 
        }
      
      processing.data.JSONArray coords  = pupilJson.getJSONArray("norm_pos"); 
      String ret = pupilJson.getInt("id") +","+ pupilJson.getDouble("timestamp") +","+ 
        (int)(coords.getDouble(0)*320.0) +","+ (int)(coords.getDouble(1)*240.0); 

      synchronized (parent) {
        String l = (now-audioSynchTime) +","+ pupilTimeStamp + "," + imuTimeStamp + "," + ret; 
        if (logData)
          pupilData.add(pupilTimeStamp + "," + imuTimeStamp + "," + ret + "," + raw); 
        lastPupil = l; 
        
      if (sendPupilConf && !sentPupilConf)
        {
        sendPupilConf = false;
        globalPupilConf = pupilConf;
        }
      if (sendPupilRate && !sentPupilRate)
        {
        sendPupilRate = false;
        globalPupilRate = pupilRate;
        }
      }

      return ret;
    }
    return null;
  }
}  