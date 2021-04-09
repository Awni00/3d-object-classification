<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>Airline Airports</title>
        <link rel="stylesheet" href="mystyle.css">
    </head>
<body>
<?php
include 'connectdb.php';
?>

<h1>Enter these Details:</h1>


<form action="addnewflight.php" method="post" class="form-inline">

<?php
$whichArrivalAirport = $_POST["newArrivalAirport"];
$whichAirportDeparted = $_POST["newAirportDeparted"];
$whichAirlineCode = $_POST["newAirlineCode"];
echo $whichAirlineCode;
echo $whichArrivalAirport; 
echo $whichAirportDeparted; 

?>

<input type= "hidden" name ="newArrivalAirport" value="<?php echo $whichArrivalAirport;?>">
<input type= "hidden" name ="newAirportDeparted" value="<?php echo $whichAirportDeparted;?>">
<input type= "hidden" name ="newAirlineCode" value="<?php echo $whichAirlineCode;?>">


            <h3>Flight Number (Enter a 3-digit number):</h3>
            <input type="text" class="form-control mb-2 mr-sm-2" name="newFlightNumber" placeholder="Flight Number">
            <div class="input-group mb-2 mr-sm-2">
            

            <h3>Select Days of Week to Offer Flight:</h3>
            Monday: <input type='checkbox' name="Monday" value="1" /></br>
            Tuesday: <input type='checkbox' name="Tuesday" value="1" /> </br>
            Wednesday: <input type='checkbox' name="Wednesday" value="1"/></br>
            Thursday: <input type='checkbox' name="Thursday" value="1"/></br>
            Friday: <input type='checkbox' name="Friday" value="1"/></br>
            Saturday: <input type='checkbox' name="Saturday" value="1"/></br>
            Sunday: <input type='checkbox' name="Sunday" value="1"/></br></br>
           
            <h3>Select Flight Plane:</h3>
           
           <?php
            $query = "SELECT * FROM Airplane";
            $result = $connection->query($query);
            echo "Which flight plane would you like to choose? </br>";
            while ($row = $result->fetch()) {
                    echo '<input type="radio" name="selectAirplane" value="';
                    echo $row["AirplaneID"];
                    echo '">' . $row["AirplaneID"] . " " . $row["AirplaneType"] . "<br>";
             }
            ?>



            <button type="submit" class="btn btn-primary mb-2">Confirm Details</button>
    </form>


</body>
</html>