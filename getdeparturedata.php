

<?php
   $query = "SELECT * FROM Flight";
   $result = $connection->query($query);
   echo "Which flight would you like to change? </br>";
   while ($row = $result->fetch()) {
        echo '<input type="radio" name="selectFlight" value="';
        echo $row["AirlineCode"];
        echo $row["FlightNumber"];
        echo '">' . $row["AirlineCode"] . " " . $row["FlightNumber"] . "<br>";
   }
?>
