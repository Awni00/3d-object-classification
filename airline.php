<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>Flight Bookings!</title>
        <link rel="stylesheet" href="mystyle.css">
  

    </head>
<body>

<element class="top-banner-slider"><element class="top-bar-info"><h1> Jada's Flights </h1></element class="top-bar-info">


<element class="header">

<element class="logo">
<ul class="top_icon">
<element class="right_cart_section">
<element class="main_bt">
<element class="top-bar-info">


<length> <percentage>

<?php
include 'connectdb.php';
?>
<element class="layout_padding">

<element class="small_main_heading">
<element class="white_main_heading"><h1> Welcome to the Flight Booking Browser</h1></element class="white_main_heading">
<element class="layout_padding">
</element class="main_bt">
<a class="readmore_bt">
<ul class="top_icon">

<li>






</br>
<img src="istockphoto-155439315-612x612.jpg" class="img-fluid">
</br>


<element class="small_main_heading"><h2>Flights Available (Arrival time, departure time)</h2>
<li>
<table class="table table-striped table-bordered table-hover table-sm">
    <thread class ="text-align:left">
        <tr>
        <th>Airline Code -</th>
        <th>Flight Number -</th>
        <th>Flight Arrival Time -</th>
        <th>Flight Departure Time</th>
        </tr>
    </thread>

<tbody>
<?php
        $query = "SELECT * FROM Flight";
        $result = $connection->query($query);
        while ($row = $result->fetch()) {
            echo "<tr>
                <td>".$row["AirlineCode"]." </td>
                <td>".$row["FlightNumber"]." </td>
                <td>".$row["ActualArrival"]."</td>
                <td>".$row["ActualDepart"]."</td>
            </tr>";
        }
?>
 <tbody>

</table>
</li>
<element class="small_main_heading"><h2>Find a Flight by Choosing an Airline Code and Flight Day</h2>

<li>
<form method="post" class="form-inline">
            <input type="text" class="form-control mb-2 mr-sm-2" name="AirlineCode" placeholder="Airline Code">
            <div class="input-group mb-2 mr-sm-2">
                <input type="text" name = "DayOffered" class="form-control" placeholder="Flight Day of Week">
            </div>

            <button type="submit" class="btn btn-primary mb-2">Submit</button>
    </br>
    </li>
    <li>
        </form>
        <?php
            if(isset($_POST['AirlineCode']) and isset($_POST['DayOffered']))
            {
                echo "
                    <table class='table table-striped table-bordered table-hover table-sm'>
                        <thead class ='text-align:left'>
                            <tr>
                            <th>Airline -</th>
                            <th>Day Offered -</th>
                            <th>Flight Number -</th>
                            <th>Departure Airport -</th>
                            <th>Arrival Airport</th>
                            </tr>
                        </thead>
    
                    <tbody>
                ";
                $AirlineCode = $_POST['AirlineCode'];
                $DayOffered = $_POST['DayOffered'];   
                //$query = 'SELECT * FROM DaysFlightOffered WHERE AirlineCode="' . $AirlineCode . '" AND DayOffered="' . $DayOffered . '" ';
                $query = 'SELECT * FROM DaysFlightOffered, Flight WHERE Flight.AirlineCode=DaysFlightOffered.AirlineCode AND Flight.FlightNumber=DaysFlightOffered.FlightNumber AND Flight.AirlineCode="' . $AirlineCode . '" AND DaysFlightOffered.DayOffered="' . $DayOffered . '" ';
                $result=$connection->query($query);
                    while ($row=$result->fetch()) {
                        echo "<tr>
                            <td>".$row["AirlineCode"]." </td>
                            <td>".$row["DayOffered"]." </td>
                            <td>".$row["FlightNumber"]." </td>
                            <td>".$row["AirportDeparted"]." </td>
                            <td>".$row["ArrivalAirport"]."</td>
                        </tr>";
                    }
                echo "</tbody></table>";
            }
        ?>
</li>
<element class="small_main_heading"><h2>Add a New Flight </h2>
<li>
<form action='getAirline.php' method="post">
<?php
   include 'getdata.php';
?>
</br>


<input type="submit"  value="Select Airline">
</form>
</li>

<element class="small_main_heading"><h2>Update Departure Time of a Flight </h2>
<li>
<form action='addNewTime.php' method="post">
<?php
   include 'getdeparturedata.php';
   //$chooseAirlineCode = $_POST['AirlineCode'];
   //$chooseFlightNumber = $_POST['FlightNumber'];  
  // echo $chooseAirlineCode; 
  // echo $chooseFlightNumberl 
 
?>
</br>
<input type="submit" name = "setNewTime" value="Select Flight">
</form>
</li>

<element class="small_main_heading"><h2>Choose a Day of the Week to find Avg. No. Seats on this Day</h2>

<form method="post" class="form-inline">
        </br>
<input type="submit"  value="Select Day">

<li>
<?php
   $query = "SELECT * FROM DaysFlightOffered GROUP BY DayOffered ";
   $result = $connection->query($query);
   echo "Which day would you like to choose? </br>";
   while ($row = $result->fetch()) {
        echo '<input type="radio" name="thisDay" value= "';
        echo $row["DayOffered"];
        echo '">' . $row["DayOffered"] . " <br>";
   }

   echo "
                    <table class='table table-striped table-bordered table-hover table-sm'>
                        <thead class ='text-align:left'>
                            <tr>
                            <th>Airline -</th>
                            <th>Day Offered -</th>
                            <th>Flight Number -</th>
                            <th>Flight Plane ID -</th>
                            <th>Number of Seats -</th>
                            </tr>
                        </thead>
    
                    <tbody>
                ";

   $thisDay = $_POST['thisDay'];
   $query = 'SELECT * FROM DaysFlightOffered, Flight WHERE Flight.AirlineCode=DaysFlightOffered.AirlineCode AND Flight.FlightNumber=DaysFlightOffered.FlightNumber AND DaysFlightOffered.DayOffered="' . $thisDay . '" ';
                    
                    $result=$connection->query($query);
                    while ($row=$result->fetch()) {
                        echo "<tr>
                            <td>".$row["AirlineCode"]." </td>
                            <td>".$row["DayOffered"]." </td>
                            <td>".$row["FlightNumber"]." </td>
                            <td>".$row["FlightPlane"]." </td>
                        </tr>";
                    }
    

    $query2 = 'SELECT * FROM AirplaneType WHERE( Flight.AirlineCode=DaysFlightOffered.AirlineCode AND Flight.FlightNumber=DaysFlightOffered.FlightNumber AND  Flight.FlightPlane = Airplane.AirplaneID AND  Airplane.AirplaneType=AirplaneType.AirplaneTypeName AND DaysFlightOffered.DayOffered="' . $thisDay . '" )';
        $result=$connection->query2($query2);
        while ($row=$result->fetch()) {
            echo "<tr>
                <td>".$row["MaxNoSeats"]."
                 </td>
        
        </tr>";          
        }
    
    
    echo "</tbody></table>";

   


?>
</li>
</form>



<?php
$connection = NULL;
?>


<element class="scrollup">

</body>
</html> 

