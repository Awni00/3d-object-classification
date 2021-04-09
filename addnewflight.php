<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>New Flight Info</title>
<link rel="stylesheet" href="mystyle.css">


</head>
<body>
<?php
   include 'connectdb.php';
?>
<h1>Your New Flight Summary Details</h1>
<br>
<ol>


<button type="submit" onclick="airline.php'/'">Click Me!</button>

<?php
            //if(isset($_POST['newAirline']) and isset($_POST['newAirportDeparted']) and isset($_POST['newArrivalAirport']) and isset($_POST['newFlightNumber'])  )
            //{
                $AirlineCode= $_POST["newAirlineCode"]; 
                $AirportDeparted = $_POST["newAirportDeparted"];
                $ArrivalAirport =$_POST["newArrivalAirport"];
                $FlightNumber = $_POST["newFlightNumber"];
                $FlightPlane = $_POST["selectAirplane"];
                echo "<br>";
                echo " Airline Code:"; 
                echo $AirlineCode ;
                echo "<br>";
                echo " Airport Departed:";
                echo $AirportDeparted;
                echo "<br>";
                echo " Arrival Airport:";
                echo $ArrivalAirport;
                echo "<br>";
                echo " Flight Number:";
                echo $FlightNumber;
                echo "<br>";
                echo "Flight Plane:";
                echo $FlightPlane; 
                echo "<br>";
                
                

                //$query0 = 'INSERT INTO Flight VALUES("' . $AirlineDeparted . '", "' . $ArrivalAirport . '","' . $AirlineCode .'" , "' . $FlightNumber . '")';
                //$query = 'INSERT INTO Flight(FlightNumber, AirlineCode, AirportDeparted, ArrivalAirport) VALUES("' . $ID . '","' . $AirlineCode . '","' . $AirportDeparted . '","' . $ArrivalAirport . '")';
                //$query2 = 'INSERT INTO DaysFlightOffered VALUES("' . $ID . '","' . $binary . '")';
                //$numRows0 = $connection->exec($query0);


                $query = 'INSERT INTO Flight values("' . $FlightNumber . '", "2020:03:10", "' . $FlightPlane . '", "' . $AirlineCode . '","' . $AirportDeparted . '","' . $ArrivalAirport . '", "00:00:00", "00:00:00", "00:00:00","00:00:00")';
                $numRows = $connection->exec($query);
                echo "Your new flight was added!";
               
            //}
        ?>




<?php 
$binary = array();
$monday = $POST_["Monday"];
echo $monday;
$days = array("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday");
foreach ($days as &$value) {
    if(isset($_POST[$value])){
        array_push($binary, 1);
    }
    else {
        array_push($binary, 0);
    }
}
$binary = implode("",$binary);
echo $binary;
$ID = rand(10,1000);

 $query2 = 'INSERT INTO DaysFlightOffered VALUES("' . $ID . '","' . $binary . '")';
$numRows2 = $connection->exec($query2);
echo "Your flight's offer days were added!";

?>





<?php


$connection = NULL;

?>

</ol>
</body>
</html>





