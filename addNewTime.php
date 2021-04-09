
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Flight Departure Time Alteration</title>
<link rel="stylesheet" href="mystyle.css">
</head>
<body>
<?php
include 'connectdb.php';
$selectFlightNumber = $_POST['selectFlight'];
?>
<element class="white_main_heading"><h1>Alter the Departure Time</h1>




<element class="small_main_heading"><h2>Enter an Updated Departure Time:</h2>
<li>

<?php
$selectFlight = $_POST['selectFlight'];
?>

<form method="post" class="form-inline">



            <input type="text" class="form-control mb-2 mr-sm-2" name="newDepartTime" placeholder="00:00:00">
            <input type= "hidden" name ="selectFlight" value="<?php echo $selectFlight;?>">
            <div class="input-group mb-2 mr-sm-2">
            </div>

            <button type="submit" class="btn btn-primary mb-2">Submit</button>

</form>


<?php
//if (isset($_POST['selectFlight'])) {

$ActualDepart = $_POST['newDepartTime'];
$selectFlight = $_POST['selectFlight'];
//$selectAirline = $_POST['chooseAirlineCode'];
//$selectFlightNumber = $_POST['selectFlight'];

echo "New Departure Time:"; 
echo $ActualDepart;
echo "</br>"; 
echo "Selected Flight :";
echo $selectFlight; 
echo "</br>"; 
$thisFlightNumber = substr($selectFlight, -3);
echo "Selected Flight Number:";
echo $thisFlightNumber;
echo "</br>"; 
echo "Selected Airline:";
$thisAirline = substr($selectFlight, 0, 2); 
echo $thisAirline;


$sql = mysql_query("UPDATE Flight SET ActualDepart = $ActualDepart WHERE (FlightNumber = $thisFlightNumber AND AirlineCode = $thisAirline)");
echo "New Flight Details Added!";



?>



</li>

<?php
$connection =- NULL;
?>
</body>