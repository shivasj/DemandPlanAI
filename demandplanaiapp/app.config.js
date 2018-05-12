(function() {
    'use strict';
    angular.module('demandplanai.app.config', [])
        .service('AppConfig', function () {
            this.Settings = {
                app_version: 1.0,
                app_id: "demandplanai",
                node_service_url: "",
                py_service_url: "http://localhost:5010",
                authentication_url: ""
            };
        });

})();
