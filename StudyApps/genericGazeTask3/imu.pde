import processing.serial.*; 

import toxi.geom.*;
import toxi.geom.mesh.*;
import toxi.math.waves.*;
import toxi.processing.*;

ToxiclibsSupport gfx;

Serial port;    // The serial port

int packetCount = 0; 
long currentSecond = 0;  

volatile String lastIMU = "";  

volatile boolean sendIMURate = false;
volatile int globalIMURate = 0;

void initIMU()
{
  gfx=new ToxiclibsSupport(this);

  String pName = getPortName("usbmodem"); 
  if (pName==null) 
  {
    println("No arduino found - error!");
    return;
  }

  println("Opening port at " + pName);
  port = new Serial(this, pName, 115200); //open 
  port.bufferUntil(10); // line feed
}


String getPortName(String search)
{
  String[] ports = Serial.list();
  for (String p : ports) 
    if (p.indexOf(search)>=0)
      return p;
  return null;
}

// we only open the connection with use IMU set to be true, so don't need check further. 
void serialEvent(Serial p) 
{ 
  long now = millis(); 
  synchronized(this) {
    imuTimeStamp = now;
  }
  String inString = p.readString();

  if (now/1000>currentSecond)
  {
    sendIMURate = true; 
    globalIMURate = (packetCount+1);
    //println("IMU Rate: " + (packetCount+1) + " Hz");
    //sendStatus("IMU Rate: " + (packetCount+1) + " Hz");
    
    currentSecond = now/1000;
    packetCount = 0;
  } else
    packetCount++; 

  synchronized (this) {
    String l = (now-audioSynchTime) +","+ pupilTimeStamp + "," + imuTimeStamp + "," + inString.substring(0, inString.length()-2);
    Matrix4x4 m = calcRot(l); 
    Vec3D vec = m.applyTo(new Vec3D(0, 200, 0));
    PVector pIntersect = Intersect(new PVector(vec.x, vec.y, vec.z), new PVector(0,0,0), new PVector(0, -1, 0), new PVector(0, 100, 0));
    l+=pIntersect.x + "," + pIntersect.z; 
    if (logData)
      imuData.add(l);
    lastIMU = l;
  }
} 

// intersection of ray and plane - needs P(osition) and D(irection) or N(ormal) of each.  
PVector Intersect(PVector rayD, PVector rayP, PVector planeN, PVector planeP)
{
  if (rayD.copy().dot(planeN)==0)
  {
    return null;
  } // parallel
  else
  {
    PVector diff  = rayP.copy().sub(planeP);
    float prod1   = diff.copy().dot(planeN);
    float prod2   = rayD.copy().dot(planeN);
    float prod3   = prod1 / prod2;
    PVector prod4 = rayD.copy().mult(prod3);
    return          rayP.copy().sub(prod4);
  }
}


Matrix4x4 calcRot(String myString)
{
  Quaternion RotQ = new Quaternion(1, 0, 0, 0);
  Matrix4x4 M1 = null;
  String inQuat[] = splitTokens(myString, ",");  
  // make sure that inQuat has a length of 4 before proceeding
  if (inQuat.length >= 7) {
    // expects I, J, K, Real from the serial port. 
    RotQ = new Quaternion(float(inQuat[6]), float(inQuat[3]), float(inQuat[4]), float(inQuat[5]));
    M1 = RotQ.toMatrix4x4();
  }
  return M1;
}



Matrix4x4 drawCube(String myString) 
{
  float qMatrix[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  Matrix4x4 m = calcRot(myString);
  if (m==null) 
    return null;
  m.toFloatArray(qMatrix);
  PMatrix M1 = getMatrix();
  
  M1.set(
    qMatrix[0], 
    qMatrix[1], 
    qMatrix[2], 
    qMatrix[3], 
    qMatrix[4], 
    qMatrix[5], 
    qMatrix[6], 
    qMatrix[7], 
    qMatrix[8], 
    qMatrix[9], 
    qMatrix[10], 
    qMatrix[11], 
    qMatrix[12], 
    qMatrix[13], 
    qMatrix[14], 
    qMatrix[15]
    );

  AABB cube;

  // Set some mood lighting
  ambientLight(128, 128, 128);
  directionalLight(128, 128, 128, 0, 0, 1);
  lightFalloff(1, 0, 0);
  lightSpecular(0, 0, 0);

  // Get to the middle of the screen
  //translate(width/4, height/4, 0);
  translate(width/2, height/2, 0);

  // Do some rotates to get oriented "behind" the device?
  rotateX(-PI/2);

  // Apply the Matrix that we generated from our IMU Quaternion
  applyMatrix(M1);

  // Draw the Cube from a 3D Bounding Box
  cube=new AABB(new Vec3D(0, 0, 0), new Vec3D(20, 20, 20));
  gfx.box(cube);
  
  return m;
}