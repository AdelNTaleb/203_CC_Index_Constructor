$(document).ready(function() {

    var my_data_line = {{ data_graph_line|safe }};
    var my_data_line_2 ={{ benchmark|safe }};
    $(function() {
       // Create the chart
       window.chart = new Highcharts.StockChart({
           plotOptions: {
                series: {
                    lineWidth: 1.5,
                   
                      states: {
                    hover: {
                        enabled: true,
                        lineWidth: 1.5
                    }
                }
                }
            },
        credits: {
            enabled:false
        },

        chart : {
            backgroundColor: null,
            plotBackgroundColor: '#fcfaf5',
            renderTo : 'chart_test'
        },
        navigator: {
            maskFill: 'rgba(251, 239, 168, 0.60)',
            series: {
                color: '#51555f',
                
            }
        },
        legend: {
            align: 'center',
            
        },
        rangeSelector : {
                 buttonTheme: { // styles for the buttons
                    fill: '#f4f4f4',
                    stroke: 'none',
                    'stroke-width': 0,

                    style: {

                        color: '#58535c',
                        fontWeight: 'bold'
                    },
                    states: {
                        hover: {
                        },
                        select: {
                            fill: '#fbefa8',
                            style: {
                                color: '#58535c'
                            }
                        }
                    // disabled: { ... }
                }
            },

            selected : 5
        },
        colors: ['#f0a800','#757b8a'],
        title : {
           text : '{{name | safe}} vs {{underlying | safe}} ',
           style:{color: "#51555f"},
       },


       series : [{
        name:"{{name | safe}}",
        data: my_data_line,
    },{
        name:"Benchmark",
        data: my_data_line_2
    }],
    tooltip: {
                pointFormat: '<span style="color:{series.color}">{series.name}</span>: <b>{point.y}</b><br/>',
                valueDecimals: 5
            },
       
        
   })
   })
    var my_data_pie = {{ data_graph_pie|safe }};
    $(function () {
        Highcharts.setOptions({
            colors: ['#ed9800', '#b22a48', '#368b99', '#47b88c', '#aac800', '#4a4647', '#676568', '#9a9899']
        });
        $('#container').highcharts({

            credits: {

                enabled:false
            },

            chart: {
                backgroundColor: null,
                type: 'pie',
                options3d: {
                    enabled: true,
                    alpha: 45
                }
            },
            title: {
                text: '{{name | safe}}\'s composition',
                style:{color: "#ffffff"},

            },
            legend:{

            itemStyle: {

                color: 'white'}},
                    plotOptions: {
                        pie: {
                            borderColor: '#ffffff',
                            innerSize: 50,
                            depth: 45,
                            dataLabels: {enabled: false},
                            showInLegend:true},
                                    style:{
                                        color:(['#ed9800', '#b22a48', '#368b99', '#47b88c', '#aac800', '#4a4647', '#676568', '#9a9899']),
                                    }
                                },
                                series: [{
                                    name: 'Weight',
                                    data: my_data_pie
                                }]
                            });
    });
});
