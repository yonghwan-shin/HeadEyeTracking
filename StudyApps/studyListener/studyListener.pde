import hypermedia.net.*;

UDP udp;  // define the UDP object

int imuRate = 0;
int pupilRate = 0;

ArrayList<Double> pupilConf = new ArrayList<Double>(); 

String imuRateHeader = "IMU rate: ";
String pupilRateHeader = "Pupil Rate: "; 
String pupilConfHeader = "Pupil confidence: ";

void setup() {
  size(300, 300); 
  textAlign(CENTER, CENTER); 
  udp = new UDP( this, 13000 );
  udp.listen( true );
}

//process events
void draw() 
  {
  background(0); 
  text("IMU Rate: " + imuRate, width/2, height/4);
  text("Pupil Rate: " + pupilRate, width/2, height/2);
  
  double ave = 0;
  for (double d: pupilConf) ave+=d;
  ave/=10.0; 
  text("Pupil Conf: " + ave, width/2, height/4*3);
  }

void receive( byte[] data, String ip, int port ) {  // <-- extended handler

  String message = new String( data );
  
  // print the result
  println( "receive: \""+message+"\" from "+ip+" on port "+port );
  if (message.substring(0, imuRateHeader.length()).equals(imuRateHeader))
    {
    imuRate = Integer.parseInt(message.substring(imuRateHeader.length()));
    }
  else if (message.substring(0, pupilRateHeader.length()).equals(pupilRateHeader))
    {
    pupilRate = Integer.parseInt(message.substring(pupilRateHeader.length()));
    }
  else if (message.substring(0, pupilConfHeader.length()).equals(pupilConfHeader))
    {
    pupilConf.add(Double.parseDouble(message.substring(pupilConfHeader.length())));
    if (pupilConf.size()>10)
      pupilConf.remove(0); 
    }
  
}