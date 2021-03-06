var canvas;
var context;
var clickX = new Array();
var clickY = new Array();
var clickDrag = new Array();
var paint = false;
var curColor = "#FFFFFF";
var offset_left = 0;
var offset_top = 0;


function drawCanvas() {

    canvas = document.getElementById('canvas');
    context = document.getElementById('canvas').getContext("2d");

    for (var o = canvas; o ; o = o.offsetParent) {
		offset_left += (o.offsetLeft - o.scrollLeft);
		offset_top  += (o.offsetTop - o.scrollTop);
    }

    $('#canvas').mousedown(function (e) {
        var mouseX = e.pageX - offset_left;
        var mouseY = e.pageY - offset_top;
        console.log(this.offsetTop)

        paint = true;
        addClick(e.pageX - offset_left, e.pageY - offset_top);
        redraw();
    });

    $('#canvas').mousemove(function (e) {
        if (paint) {
            addClick(e.pageX - offset_left, e.pageY - offset_top, true);
            redraw();
        }
    });

    $('#canvas').mouseup(function (e) {
        paint = false;
    });
}

function clearCanvas(){
    clickX = new Array();
    clickY = new Array();
    clickDrag = new Array();
    context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas
}

function addClick(x, y, dragging) {
    clickX.push(x);
    clickY.push(y);
    clickDrag.push(dragging);
}

function redraw() {
    
    context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas
    context.strokeStyle = curColor;
    context.lineJoin = 'round';
    context.lineWidth = 15;

    for (var i = 0; i < clickX.length; i++) {
        context.beginPath();

        if (clickDrag[i] && i) {
            context.moveTo(clickX[i - 1], clickY[i - 1]);
        } else {
            context.moveTo(clickX[i] - 1, clickY[i]);
        }
        context.lineTo(clickX[i], clickY[i]);
        context.closePath();
        context.stroke();
    }
}

function save() {
    var image = new Image();
    var url = document.getElementById('url');
    image.id = "pic";
    image.src = canvas.toDataURL();
    url.value = image.src
}
