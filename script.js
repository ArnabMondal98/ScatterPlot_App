let myChart = null;

function updateChart() {
    const topic = document.getElementById('topicSelect').value;
    if (!topic) return;

    fetch(`/get_data/${topic}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }

            const ctx = document.getElementById('scatterChart').getContext('2d');
            
            // Define colors for different stratifications
            const uniqueStratifications = [...new Set(data.stratifications)];
            const colors = [
                'rgba(75, 192, 192, 0.6)',
                'rgba(255, 99, 132, 0.6)',
                'rgba(54, 162, 235, 0.6)',
                'rgba(255, 206, 86, 0.6)',
                'rgba(153, 102, 255, 0.6)',
                'rgba(255, 159, 64, 0.6)'
            ];
            const stratificationColors = {};
            uniqueStratifications.forEach((strat, i) => {
                stratificationColors[strat] = colors[i % colors.length];
            });

            // Prepare datasets
            const datasets = uniqueStratifications.map(strat => ({
                label: strat,
                data: data.x.map((x, i) => {
                    if (data.stratifications[i] === strat) {
                        return { x: x, y: data.y[i], label: data.labels[i] };
                    }
                    return null;
                }).filter(point => point !== null),
                backgroundColor: stratificationColors[strat],
                borderColor: stratificationColors[strat].replace('0.6', '1'),
                pointRadius: 6,
                pointHoverRadius: 8
            }));

            // Destroy existing chart if it exists
            if (myChart) {
                myChart.destroy();
            }

            // Create new chart
            myChart = new Chart(ctx, {
                type: 'scatter',
                data: { datasets },
                options: {
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: data.x_label,
                                font: { size: 14 }
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: data.y_label,
                                font: { size: 14 }
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const point = context.raw;
                                    return `${point.label}: (${point.x}, ${point.y})`;
                                }
                            }
                        },
                        legend: {
                            position: 'top'
                        }
                    }
                }
            });
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while fetching data');
        });
}