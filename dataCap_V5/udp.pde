import hypermedia.net.*;

UDP udp;     // define the UDP object

String UDP_START = "";

void initUDP()
  {
  // create a new datagram connection on port 12000 and wait for incomming message
  udp = new UDP( this, 12001 );
  //udp.log( true );     // <-- printout the connection activity
  udp.listen( true );
  }
  

void sendMsg(String msg)
  {
  if (remoteIP.equals("N/A"))
    {println("No remote IP found"); return;}
    
  String ip       = remoteIP;     // "127.0.0.1";  // the remote IP address
  int port        = 30000;        // the destination port
    
  //println(msg, ip, port);   
  udp.send( msg, ip, port );
  }
  

void sendStatus(String msg)
  {
  if (statusIP.equals("N/A"))
    {println("No status IP found"); return;}
    
  String ip       = statusIP;     // "127.0.0.1";  // the remote IP address
  int port        = 13000;        // the destination port
    
  //println(msg, ip, port);
  udp.send( msg, ip, port );
  }

  
void receive(byte[] data, String ip, int port) {  
  String msg = new String( data );
  UDP_START = msg;
  println( "receive: \""+msg+"\" from "+ip+" on port "+port );
}



/*
 * Android + OSX version is below
 * from : http://stackoverflow.com/questions/6064510/how-to-get-ip-address-of-the-device
 * works on my (single) osx install. Give two address on my (single) android test device
 * Get IP address from first non-localhost interface
 * @param ipv4  true=return ipv4, false=return ipv6
 * @return  address or empty string
 */
import java.net.NetworkInterface; 
import java.net.InetAddress;
import java.util.*;        // for collections - we shuffle the faces for randomisation

public static String getIPAddress() {return getIPAddress(true);} 
public static String getIPAddress(boolean useIPv4) 
{
  String totalAddresses = "IPs: "; 
  int count = 0;
  try {
    List<NetworkInterface> interfaces = Collections.list(NetworkInterface.getNetworkInterfaces());
    for (NetworkInterface intf : interfaces) {
      List<InetAddress> addrs = Collections.list(intf.getInetAddresses());
      for (InetAddress addr : addrs) {
        if (!addr.isLoopbackAddress()) {
          String sAddr = addr.getHostAddress();
          //boolean isIPv4 = InetAddressUtils.isIPv4Address(sAddr);
          boolean isIPv4 = sAddr.indexOf(':')<0;

          if (useIPv4) {
            // "mnet" is typically used as part of the mobile network adaptor not wifi. CHECK THIS ON YOUR DEVICE!
            if (isIPv4 && intf.getDisplayName().indexOf("mnet")==-1) 
              {
              if (totalAddresses.length()>5)
                totalAddresses += ", "; 
              totalAddresses += sAddr;// + "-" + intf.getDisplayName(); //return sAddr;
              count ++; 
              }
          } else {
            if (!isIPv4) {
              int delim = sAddr.indexOf('%'); // drop ip6 zone suffix
              return delim<0 ? sAddr.toUpperCase() : sAddr.substring(0, delim).toUpperCase();
            }
          }
        }
      }
    }
  } 
  catch (Exception ex) {
  } // for now eat exceptions
  return totalAddresses + " (" + count + ")";
}  
