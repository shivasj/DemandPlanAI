(function(window) {
    'use strict';
    var d3components = {
        demandChart : demandChart
    }

    // Demand Chart
    function demandChart() {
        // Variables
        var margin = { top: 0, right: 0, bottom: 0, left: 0 },
            width = 500,
            height = 300,
            duration = 500,
            dataValue = function(d) { return +d.data; },
            responsive = true,
            fixedHeight = false,
            displayXAxis = true;

        var _selection;
        var bottomOffset = 30;

        // Chart creation function
        function chart(selection) {
            _selection = selection;

            // For each selection
            _selection.each(function(data) {
                // Build the visual
                buildVisual(this,data);
            });

        }
        function buildVisual(handle,data1) {

            // Check if responsive
            if (responsive) {
                // Get dimension from container
                width = parseInt(_selection.style("width"));
                if(!fixedHeight){
                    height = parseInt(width * 0.30);
                }
            }

            if(!displayXAxis){
                bottomOffset =0;
            }

            var data = angular.copy(data1)

            // parse the date / time
            var parseTime = d3.timeParse("%m/%d/%Y");
            // Prepare data
            data.history.forEach(function(d) {
                d.date = parseTime(d.date);
                d.qty = +d.qty;
            });
            data.prediction.forEach(function(d) {
                d.date = parseTime(d.date);
                d.qty = +d.qty;
            });
            console.log(data)

            var x1width = width*0.8;
            var x2width = width*0.2;
            // Scales
            var t = d3.transition().duration(duration).ease(d3.easeLinear);
            var x1 = d3.scaleTime().rangeRound([0, x1width-margin.left]);
            var x2 = d3.scaleTime().rangeRound([x1width-margin.left, width-margin.right]);
            var y = d3.scaleLinear().rangeRound([height-margin.top-margin.bottom-bottomOffset, 0]);

            var dateFormat = d3.timeFormat("%b %Y");
            var dateFormat2 = d3.timeFormat("%d %b");

            // Line
            var line = d3.line()
                .curve(d3.curveBasis)
                .x(function(d) { return x1(d.date); })
                .y(function(d) { return y(d.qty); });
            var line2 = d3.line()
                .curve(d3.curveBasis)
                .x(function(d) { return x2(d.date); })
                .y(function(d) { return y(d.qty); });

            // Set Domain of x and y
            var x1Min = d3.min(data.history, function(d) { return d.date; });
            var x1Max = d3.max(data.history, function(d) { return d.date; });
            var x2Min = d3.min(data.prediction, function(d) { return d.date; });
            var x2Max = d3.max(data.prediction, function(d) { return d.date; });
            var yMin = d3.min(data.history, function(d) { return d.qty; });
            var yMax = d3.max(data.history, function(d) { return d.qty; });
            //yMax = yMax*1.1;
            x1.domain([x1Min, x1Max]);
            x2.domain([x2Min, x2Max]);
            y.domain([yMin, yMax]);

            function tick() {
                d3.select(this)
                    .attr("d", function(d) {
                        return line(d);
                    })
                    .attr("transform", null);
            }
            function tick2() {
                d3.select(this)
                    .attr("d", function(d) {
                        return line2(d);
                    })
                    .attr("transform", null);
            }

            // Chart Building Logic
            // ***************************************************************
            // Initialize components
            var svg = d3.select(handle).selectAll("svg").data([0]);
            var gEnter = svg.enter().append("svg").append("g");

            // Axis
            gEnter.append("g").attr("class", "axis x");
            gEnter.append("g").attr("class", "axis x2");
            gEnter.append("g").attr("class", "axis y");

            // Line 1
            gEnter.append("g")
                .attr("class", "history-lines")
                .selectAll(".data")
                .data([0])
                .enter()
                .append("path")
                .attr("class", "data");

            // Line 2
            gEnter.append("g")
                .attr("class", "prediction-lines")
                .selectAll(".data")
                .data([0])
                .enter()
                .append("path")
                .attr("class", "data");


            // Define dimensions
            var svg = _selection.select("svg");
            svg.attr('width', width).attr('height', height);
            var g = svg.select("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

            // Add the x Axis
            if(displayXAxis){
                g.select("g.axis.x")
                    .attr("transform", "translate(0," + (height-margin.bottom-margin.top-bottomOffset) + ")")
                    .transition(t)
                    .call(d3.axisBottom(x1).tickFormat(dateFormat))
                    .select(".domain")
                    .remove();

                // x Axis text format
                g.selectAll("g.axis.x text")
                    .style("stroke-width", 1)
                    .style('fill','#000');

                g.select("g.axis.x2")
                    .attr("transform", "translate(0," + (height-margin.bottom-margin.top-bottomOffset) + ")")
                    .transition(t)
                    .call(d3.axisBottom(x2).tickFormat(dateFormat2).ticks(3))
                    .select(".domain")
                    .remove();

                // x Axis text format
                g.selectAll("g.axis.x2 text")
                    .style("stroke-width", 1)
                    .style('fill','#000');
            }
            // y Axis
            g.select("g.axis.y")
                .call(d3.axisLeft(y))
                .append("text")
                .attr("fill", "#000")
                .attr("transform", "rotate(-90)")
                .attr("y", 6)
                .attr("dy", "0.71em")
                .attr("text-anchor", "end")
                .text("Quantity (Tho)");

            // Path
            g.selectAll("g.history-lines path.data")
                .data([data.history])
                .attr("stroke", "steelblue")
                .attr("stroke-linejoin", "round")
                .attr("stroke-linecap", "round")
                .attr("stroke-width", 1.5)
                .style("fill", "none")
                .transition()
                .duration(duration)
                .ease(d3.easeLinear)
                .on("start", tick);

            g.selectAll("g.prediction-lines path.data")
                .data([data.prediction])
                .attr("stroke", "red")
                .attr("stroke-linejoin", "round")
                .attr("stroke-linecap", "round")
                .attr("stroke-width", 1.5)
                .style("fill", "none")
                .transition()
                .duration(duration)
                .ease(d3.easeLinear)
                .on("start", tick2);


            // ***************************************************************
            // Chart Building Logic
        }

        // Define Getters and Setters
        chart.margin = function(_) {
            if (!arguments.length) return margin;
            margin = _;
            return chart;
        };
        chart.width = function(_) {
            if (!arguments.length) return width;
            width = _;
            return chart;
        };
        chart.height = function(_) {
            if (!arguments.length) return height;
            height = _;
            return chart;
        };
        chart.dataValue = function (_) {
            if (!arguments.length) return dataValue;
            dataValue = _;
            return chart;
        };
        chart.responsive = function(_) {
            if (!arguments.length) return responsive;
            responsive = _;
            return chart;
        };
        chart.displayCurrentPoint = function(_) {
            if (!arguments.length) return displayCurrentPoint;
            displayCurrentPoint = _;
            return chart;
        };
        chart.fixedHeight = function(_) {
            if (!arguments.length) return fixedHeight;
            fixedHeight = _;
            return chart;
        };
        chart.displayXAxis = function(_) {
            if (!arguments.length) return displayXAxis;
            displayXAxis = _;
            return chart;
        };

        // Resize function
        chart.resize = function () {
            // For each selection
            _selection.each(function(data) {
                // Build the visual
                buildVisual(this,data);
            });
        }

        // return the chart function
        return chart;
    }
    // Demand Chart

    window.d3components =d3components;
})(window);