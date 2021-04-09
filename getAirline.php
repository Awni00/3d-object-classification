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



<h1>Let's select from the Available Airports:</h1>


<?php
//$newAir = $_POST["newAirlineCode"];
$whichAirlineCode = $_POST["newAirlineCode"];
echo $whichAirlineCode;

?>


<form action="getAirports.php" method="post">

<input type= "hidden" name ="newAirlineCode" value="<?php echo $whichAirlineCode;?>">

<?php 
   include 'getairportdata.php';
?>
</br> 


<input type="submit" value="Save this flight info">
</form>

</br>


<?php
   $connection = NULL;
?>


</body>
</html>