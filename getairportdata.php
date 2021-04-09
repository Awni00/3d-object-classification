<?php
   $query = "SELECT * FROM Airport ";
   $result = $connection->query($query);
   echo "Which arrival airport would you like to choose? </br>  <br>";
  
   while ($row = $result->fetch()) {
        echo '<input type="radio" name="newArrivalAirport" value= "';
        echo $row["AirportCode"];
        echo '">' . $row["AirportCode"] ." ". $row["AirportName"] . " <br>  ";

   }

   $query = "SELECT * FROM Airport ";
   $result = $connection->query($query);
   echo " </br> Which departure airport would you like to choose? </br>  <br>";

   while ($row = $result->fetch()) {
        echo '<input type="radio" name="newAirportDeparted" value= "';
        echo $row["AirportCode"];
        echo '">' . $row["AirportCode"] ." ". $row["AirportName"] . " <br>  ";



  
   }

  
   // $ChosenAirlineCode = $_POST['AirlineCode']; 
    //$query = 'SELECT * FROM Airplane WHERE AirlineCode="' . $AirlineCode . '" ';
    //$result=$connection->query($query);
      //      while ($row=$result->fetch()) {
        //        echo '<input type="radio" name="flight" value= "';
         //       echo $row["AirplaneType"];
         //       echo '">' . $row["AirplaneType"] . " <br>";
      //      }

?>

