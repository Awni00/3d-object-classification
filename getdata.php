
<?php
   $query = "SELECT * FROM Flight GROUP BY AirlineCode";
   $result = $connection->query($query);
   echo "Which airline would you like to add a flight to? </br>";
   while ($row = $result->fetch()) {
        echo '<input type="radio" name="newAirlineCode" value= "';
        echo $row["AirlineCode"];
        echo '">' . $row["AirlineCode"] . " <br>";
   }
?>