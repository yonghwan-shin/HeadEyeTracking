/*
  Using the BNO080 IMU
  By: Nathan Seidle
  SparkFun Electronics
  Date: December 21st, 2017
  License: This code is public domain but you buy me a beer if you use this and we meet someday (Beerware license).

  Feel like supporting our work? Buy a board from SparkFun!
  https://www.sparkfun.com/products/14586

  This example shows how to output the i/j/k/real parts of the rotation vector.
  https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

  It takes about 1ms at 400kHz I2C to read a record from the sensor, but we are polling the sensor continually
  between updates from the sensor. Use the interrupt pin on the BNO080 breakout to avoid polling.

  Hardware Connections:
  Attach the Qwiic Shield to your Arduino/Photon/ESP32 or other
  Plug the sensor onto the shield
  Serial.print it out at 9600 baud to serial monitor.
*/

#include <Wire.h>

#include "SparkFun_BNO080_Arduino_Library.h"
BNO080 myIMU;



int ptsAfterZero = 5;

void setup()
{
  Serial.begin(9600); //  for mkr1010, baud rate is not needed - runs at max it supports always
  Serial.println();
  Serial.println("BNO080 Read Example");

  Wire.begin();

  if (myIMU.begin() == false)
  {
    Serial.println("BNO080 not detected at default I2C address. Check your jumpers and the hookup guide. Freezing...");
    while (1);
  }

  Wire.setClock(400000); //Increase I2C data rate to 400kHz
//  Wire.setClock  (1000000); //Increase I2C data rate to 400kHz

  myIMU.enableRotationVector(5); //Send data update every 50ms

  Serial.println(F("Rotation vector enabled"));
  Serial.println(F("Output in form i, j, k, real, accuracy"));
  
}

void loop()
{
  
  //Look for reports from the IMU
  if (myIMU.dataAvailable() == true)
  {
    float quatI = myIMU.getQuatI();
    float quatJ = myIMU.getQuatJ();
    float quatK = myIMU.getQuatK();
    float quatReal = myIMU.getQuatReal();
    float quatRadianAccuracy = myIMU.getQuatRadianAccuracy();

    
    Serial.print(quatI, ptsAfterZero);
    Serial.print(F(","));
    Serial.print(quatJ, ptsAfterZero);
    Serial.print(F(","));
    Serial.print(quatK, ptsAfterZero);
    Serial.print(F(","));
    Serial.print(quatReal, ptsAfterZero);
    Serial.print(F(","));
    Serial.print(quatRadianAccuracy, ptsAfterZero);
    Serial.print(F(","));

    Serial.println();
  }
}
