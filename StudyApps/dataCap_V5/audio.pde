import processing.sound.*;

micThread m; 
long localPingTime=-1; 
private volatile long pingTime = -1;

// open the micThread
void initAudio() {
  m = new micThread(this); 
  m.start(15000);
}

// start listening for pinging - we don't do this all the time to save cycles
void listenAudio()
  {
  println("Audio: starting to listen...");
  synchronized(this) {m.startListen();} 
  }


// check for pinging
long checkAudio() {
    synchronized(this) {
      localPingTime = pingTime; 
      pingTime = -1;
      }
    if (localPingTime!=-1) 
      return localPingTime;
    return -1;
}

  
  
  



class micThread extends Thread {  
  private volatile boolean quit = false;
  private volatile boolean listen = true;

  PApplet parent; 

  AudioIn input;
  FFT fft;
  int bands;   // nuymber of bands we use
  int targetBand;
  
  micThread(PApplet parIn)
  {
    parent = parIn;
  }

  public void start(int tB)
  {
    targetBand = tB; 
    listen = false;
    // Create an Audio input and grab the 1st channel
    input = new AudioIn(parent, 0);

    // Create at FFT on the input
    bands = 16; // los bands for less math? 
    fft = new FFT(parent, bands);
    fft.input(input);

    // Begin capturing the audio input
    input.start();

    super.start();
  }

  public void startListen()
  {
    listen = true;
  }

  public void stopListen()
  {
    listen = false;
  }

  public void quit()
  {
    quit = true;
  }

  public void run()
  {
    println("Mic reading thread starting up.");

    while (!quit) {
      if (listen)
      {
        fft.analyze();
        int peakBand = -1;
        float peakBandVal = -100; 
        for (int i = 0; i < bands; i++) {
          if (fft.spectrum[i] > peakBandVal)
          {
            peakBand = i;
            peakBandVal = fft.spectrum[i];
          }
        }

        float f = (peakBand * 22050) / bands;
        if (peakBandVal>0.01 && f>=targetBand)
        {
          synchronized(parent) {pingTime = millis();} 
          listen = false;
        }
        else
          delay(1);
      }
      else
        delay(10); 
    }

    println("Mike scanning thread shutting down.");
  }
}  